import logging
import os

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from dataloading import PAD_IDX, SOS_IDX
from utils import tighten

logger = logging.getLogger(__name__)

MAXLEN = 25

# TODO: 'sos' and 'eos' in encoder?
# TODO: dropout in embedding?
# TODO: handle when bidrectional
class Seq2Seq(nn.Module):
    def __init__(self, name, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.name = name
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, y):
        """
        input
            x: (B, L)
            x_lengths: (B)
            y: (B, L')
            y_lengths: (B)

        return
            logits: (B, L', num_embeddings)
            attn_weights: (B, L', L)
        """
        memory, final_hidden = self.encoder(x)
        logits, attn_weights = self.decoder(y, memory, final_hidden)
        return logits, attn_weights

    def generate(self, x):
        memory, final_hidden = self.encoder(x)
        generated = self.decoder.generate(memory, final_hidden)
        return generated


class Encoder(nn.Module):
    def __init__(self, embedding, hidden_size, num_layers, dropout,
                 bidirectional=False):
        super(Encoder, self).__init__()
        self.embedding = embedding
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True,
                            bidirectional=bidirectional)

    def forward(self, x):
        """
        input
            x: ((B, L), (B,))

        return
            memory: (B, L, num_directions*hidden_size)
            memory_mask: (B, L)
            final_hidden: (num_layers*num_directions, B, hidden_size)
        """
        x, lengths = x
        total_length = x.size(1) # for dataparallel
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True)
        memory, final_hidden = self.lstm(packed)
        memory, _ = pad_packed_sequence(memory, batch_first=True,
                                        total_length=total_length)
        memory_mask = (x == PAD_IDX)
        return (memory, memory_mask), final_hidden


class Decoder(nn.Module):
    def __init__(self, embedding, attention, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.embedding = embedding
        self.attention = attention
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.out = nn.Linear(in_features=hidden_size,
                             out_features=self.embedding.num_embeddings)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, y, memory, final_hidden):
        """
        input
            y: ((B, L'), (B,))
            memory:  (B, L, num_directions*hidden_size)
            memory_mask: (B, L)
            final_hiddem:  (num_layers*num_directions, B, hidden_size)

        return
            logits: (B, L', num_embeddings)
            attn_weights: (B, L', L)
        """
        y, _ = y
        embedded = self.embedding(y)
        dec_outputs, _ = self.lstm(embedded, final_hidden)
        memory, memory_mask = memory
        attn_outputs, attn_weights = self.attention(query=dec_outputs.transpose(0, 1),
                                                    key=memory.transpose(0, 1),
                                                    value=memory.transpose(0, 1),
                                                    key_padding_mask=memory_mask)
        logprobs = self.logsoftmax(self.out(attn_outputs.transpose(0, 1)))
        return logprobs, attn_weights

    def generate(self, memory, final_hidden):
        B = memory.size(0)
        memory, memory_mask = memory
        input_token = final_hidden.new_full((B, 1), SOS_IDX)
        hidden = final_hidden
        generated = torch.empty(B, MAXLEN)
        for t in range(MAXLEN):
            input_embedded = self.embedding(input_token)
            dec_output, hidden = self.lstm(input_embedded, hidden)
            attn_output, _ = self.attention(query=dec_output.transpose(0, 1),
                                                  key=memory.transpose(0, 1),
                                                  value=memory.transpose(0, 1),
                                                  key_padding_mask=memory_mask)
            logprobs = self.logsoftmax(self.out(attn_output.squeeze()))
            _, generated_token = logprobs.topk(1) # greedy decoding
            generated[t] = generated_token
            input_token = generated_token
        return tighten(generated)



def build_model(name, device, vocab_size, embedding_size, hidden_size, num_layers,
                dropout, bidrectional=False, embedding_weights=None):
    if embedding_weights is None:
        embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD_IDX)
    else: # freezed by default
        embedding = nn.Embedding.from_pretrained(embedding_weights)
    encoder = Encoder(embedding, hidden_size, num_layers, dropout, bidrectional)
    attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1)
    decoder = Decoder(embedding, attention, hidden_size, num_layers, dropout)
    model = Seq2Seq(name, encoder, decoder)
    logging.info('building model...{}\n'.format(name) + str(model))
    return model.to(device)


def load_model(path, *args, **kwargs):
    model = Seq2Seq(*args, **kwargs)
    logger.info('loading model from... {}'.format(path))
    return model.load_state_dict(torch.load(path))


# TODO: save_checkpoint like in DME code
def save_model(model, savedir, filename):
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    filename = model.name + '_{:.2}.pt'.format(filename)
    savedir = os.path.join(savedir, filename)
    torch.save(model.state_dict(), savedir)
    logger.info('saving model in {}'.format(savedir))
    return savedir


if __name__ == '__main__':
    seq2seq = build_model('standard', 30000, 300, 500, 1, 0.3)
    print(seq2seq)
