import os
import logging

import torch
# import matplotlib.pyplot as plt

from dataloading import EOS_IDX, SOS_IDX, UNK_IDX, PAD_IDX

logger = logging.getLogger(__name__)

### data related
def truncate(x, token=None):
    # delete a special token in a batch
    assert token in ['sos', 'eos'], 'can only truncate sos or eos'
    x, lengths = x # (B, L)
    lengths = lengths - 1
    if token == 'sos': x = x[:, 1:]
    elif token == 'eos': x = x[:, :-1]
        # x[x==EOS_IDX] = PAD_IDX # unnecessary when ignore_index is set
    return (x, lengths)

def append(x, token=None):
    # add a special token to a batch
    assert token in ['sos', 'eos'], 'can only append sos or eos'
    x, lengths = x # (B, L), (B,)
    lengths += 1
    B = x.size(0)
    if token == 'eos':
        eos = x.new_full((B,1), EOS_IDX)
        x = torch.cat([x, eos], dim=1)
    elif token == 'sos':
        sos = x.new_full((B,1), SOS_IDX)
        x = torch.cat([sos, x], dim=1)
    return (x, lengths)

def sequence_mask(lengths):
    # make a mask matrix corresponding to given length
    # from https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/ops/array_ops.py
    row_vector = torch.arange(0, max(lengths), device=lengths.device) # (L,)
    matrix = lengths.unsqueeze(-1) # (B, 1)
    result = row_vector < matrix # 1 for real tokens
    return result # (B, L)

def sort_by_length(y, lengths):
    # sort hy, y in decreasing order(for compatibility with packed_sequence)
    l = [(sent, l) for sent, l in zip(y, lengths)]
    sorted_l = sorted(l, key=lambda x: x[1], reverse=True)
    h = torch.stack([i[0] for i in sorted_l], dim=0)
    lengths = torch.stack([i[1] for i in sorted_l], dim=0)
    return h, lengths

def get_actual_lengths(y):
    # get actual length of a generated batch considering eos
    B = y.size(0)
    non_zeros = (y == EOS_IDX).nonzero()
    is_dirty = [False for _ in range(B)]
    lengths = []
    for idx in non_zeros:
        i, j = idx[0].item(), idx[1].item()
        if not is_dirty[i]:
            is_dirty[i] = True
            lengths.append(j+1) # zero-index
    return torch.tensor(lengths, device=non_zeros.device)

def tighten(y):
    """
    pad tokens after EOS and mask hiddens after EOS
    y: (B, MAXLEN+1)
    """
    def replace(y):
        y[:, -1] = EOS_IDX
        return y
    y = replace(y) # some sents don't have eos at all
    lengths = get_actual_lengths(y)
    mask = sequence_mask(lengths)
    y = y[:, :mask.size(1)]  # truncate unnecessarily generated part
    y.masked_fill_((mask != 1), PAD_IDX)  # this does not backprop
    y, lengths = sort_by_length(y, lengths)
    return y, lengths

#def sort_by_length(y, lengths):
#    # sort hy, y in decreasing order(for compatibility with packed_sequence)
#    l = [(tok, l) for tok, l in zip(y, lengths)]
#    sorted_l = sorted(l, key=lambda x: x[1], reverse=True)
#    h = torch.stack([i[0] for i in sorted_l], dim=0)
#    lengths = torch.stack([i[1] for i in sorted_l], dim=0)
#    return h, lengths

def reverse(batch, vocab):
    # turn a batch of idx to tokens
    batch = batch.tolist()
    def trim(s, t):
        sentence = []
        for w in s:
            if w == t:
                break
            sentence.append(w)
        return sentence
    batch = [trim(ex, EOS_IDX) for ex in batch]
    batch = [' '.join([vocab.itos[i] for i in ex]) for ex in batch]
    return batch

## TODO: various experiment with uuid
#def write_to_file(write_list, msg, data_type, epoch, savedir='experiment'):
#    if not os.path.isdir(savedir): os.mkdir(savedir)
#    filename = '{}/{}_epoch{}'.format(savedir, data_type, epoch)
#    with open(filename, 'w') as f:
#        for to_write in write_list:
#            for orig, summ, ref in to_write:
#                f.write('===== orig =====\n' + '\n'.join(orig) + '\n')
#                f.write('===== generated summmary =====\n' + summ + '\n')
#                f.write('===== reference ====\n' + ref + '\n\n')
#        f.write(msg)
#
#
## TODO decorator?
#def plot_wrapper(plot_func, *args, filename=None):
#    def ret_func():
#        plt.figure()
#        plot_func(*args)
#        plt.legend()
#        plt.savefig(filename)
#    return ret_func
#
#def plot_kl_loss(kl_stats, filename='kl_stats.png'):
#    steps = range(len(kl_stats))
#    plt.figure(1)
#    plt.plot(steps, kl_stats, label='kl_loss')
#    plt.plot(steps, [kl_coef(i) for i in steps], label='kl_coef')
#    plt.xlabel('Step')
#    plt.ylabel('KL loss')
#    plt.legend()
#    plt.savefig(filename)
#    logger.info('kl_loss graph saved at {}'.format(filename))
#
#def plot_learning_curve(train_losses, valid_losses, filename='learning_curve.png'):
#    epochs = range(1, len(train_losses)+1, 1)
#    plt.figure(2)
#    plt.plot(epochs, train_losses, label='train_loss')
#    plt.plot(epochs, valid_losses, label='valid_loss')
#    plt.xlabel('Step')
#    plt.ylabel('Recon loss + KL loss')
#    plt.legend()
#    plt.savefig(filename)
#    logger.info('learning curve saved at {}'.format(filename))
#
#def plot_metrics(train_metrics, valid_metrics, filename='metrics.png'):
#    epochs = range(1, len(train_metrics)+1, 1)
#    train_bleu_1 = [m['Bleu_1'] for m in train_metrics]
#    train_meteor = [m['METEOR'] for m in train_metrics]
#    valid_bleu_1 = [m['Bleu_1'] for m in valid_metrics]
#    valid_meteor = [m['METEOR'] for m in valid_metrics]
#    plt.figure(3)
#    plt.plot(epochs, train_bleu_1, label='train_bleu_1')
#    plt.plot(epochs, train_meteor, label='train_meteor')
#    plt.plot(epochs, valid_bleu_1, label='valid_bleu_1')
#    plt.plot(epochs, valid_meteor, label='train_meteor')
#    plt.xlabel('Epoch')
#    plt.ylabel('Performance')
#    plt.legend()
#    plt.savefig(filename)
#    logger.info('metrics graph saved at {}'.format(filename))
