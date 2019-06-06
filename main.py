import logging

from setproctitle import setproctitle

import torch

from dataloading import Data
from model import build_model
from rewards import get_mutual_information, get_reward_funcs
from trainer import build_trainer

setproctitle("(hwijeen) RL dialogue")
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


# TODO: argparse
if  __name__ == "__main__":
    DATA_DIR = 'data/'
    DEVICE = torch.device('cuda:0')
    EPOCH = 10
    TRAIN_MODE = 'SUPERVISED_PRETRAIN' 
    MUTUAL_INFORMATION = False

    EMBEDDING_SIZE = 300
    HIDDEN_SIZE = 500
    NUM_LAYERS = 1
    DROPOUT = 0.3

    data = Data(DATA_DIR, DEVICE, batch_size=64, use_glove=False)

    ####################
    from utils import reverse

    path = 'models/forward_0.95.pt'
    model = build_model(name='forward',
                          device=DEVICE,
                          vocab_size=len(data.vocab),
                          embedding_size=EMBEDDING_SIZE,
                          hidden_size=HIDDEN_SIZE,
                          num_layers=NUM_LAYERS,
                          dropout=DROPOUT)
    model.load_state_dict(torch.load(path))
    for batch in data.valid_iter:
        generated, lengths = model.generate(batch.merged_hist)
        print(reverse(generated, data.vocab))
        input()
   #################################

    if TRAIN_MODE == 'SUPERVISED_PRETRAIN':
        seq2seq = build_model(name='forward',
                              device=DEVICE,
                              vocab_size=len(data.vocab),
                              embedding_size=EMBEDDING_SIZE,
                              hidden_size=HIDDEN_SIZE,
                              num_layers=NUM_LAYERS,
                              dropout=DROPOUT)
        trainer = build_trainer(kind='supervised',
                                model=seq2seq,
                                data=data,
                                lr=1e-3,
                                lr_shrink=2e-1,
                                scheduler_patience=3,
                                clip=5,
                                records=['NLLLoss'])
        results = trainer.train(num_epoch=EPOCH, verbose=True)
        seq2seq_back = build_model(name='backward',
                                   device=DEVICE,
                                   vocab_size=len(data.vocab),
                                   embedding_size=EMBEDDING_SIZE,
                                   hidden_size=HIDDEN_SIZE,
                                   num_layers=NUM_LAYERS,
                                   dropout=DROPOUT)
        trainer_back = build_trainer(kind='supervised',
                                     model=seq2seq_back,
                                     data=data,
                                     lr=1e-3,
                                     lr_shrink=2e-1,
                                     scheduler_patience=3,
                                     clip=5,
                                     records=['NLLLoss'],
                                     backward=True)
        results_back = trainer_back.train(num_epoch=EPOCH, verbose=True)

#    if MUTUAL_INFORMATION is None:
#        seq2seq_rl = Seq2Seq.load(SUPERVISED_FORWARD, VOCAB_SIZE, EMBEDDING,
#                                  HIDDEN, name='mutual information')
#        mi = get_mutual_information(SUPERVISED_FORWARD, SUPERVISED_BACKWARD,
#                                        VOCAB_SIZE, EMBEDDING, HIDDEN)
#        trainer_rl = RLTrainer(seq2seq_rl, data, mi, lr=0.001, clip=5,
#                               records=['Mutual Informtion'])
#
#    else:
#        seq2seq_rl = Seq2Seq.load(MUTUAL_INFORMATION, VOCAB_SIZE, EMBEDDING,
#                                  HIDDEN, name='RL')
#        minfo, eanswer, iflow = get_reward_funcs(SUPERVISED_FORWARD,
#                                                 SUPERVISED_BACKWARD, VOCAB_SIZE,
#                                                 EMBEDDING, HIDDEN)
#        trainer_RL = RLTrainer(seq2seq_rl, data, (minfo, eanswer, iflow),
#                               lr=0.001, clip=5, records=['Mutual Information',
#                                                          'Information Flow'
#                                                          'Ease of Answering'])



