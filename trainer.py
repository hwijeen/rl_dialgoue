import logging
from abc import abstractmethod

import torch.nn as nn
import torch.optim as optim
from nlgeval import NLGEval

from utils import truncate, reverse
from dataloading import PAD_IDX
from train_utils import Stats, EarlyStopper
from model import load_model, save_model

logger = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(self, model, data, optimizer, scheduler, clip, stats, savedir):
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip = clip
        self.stats = stats
        self.savedir = savedir

    @abstractmethod
    def _compute_loss(self):
        pass

    @abstractmethod
    def _run_epoch(self):
        pass

    # TODO: early stopping
    def train(self, num_epoch, verbose=True):
        for epoch in range(1, num_epoch+1, 1):
            loss = self._run_epoch(verbose)
        savedir = save_model(self.model, self.savedir, loss)
        return {'savedir': savedir, 'throughout': self.stats.throughout,
                'running_avg': self.stats.running_avg}

    def generate(self, path, *args, **kwargs):
        model =  load_model(path, *args, **kwargs)
        for batch in self.data.valid_iter:
            generated = model.generate(batch.merged_hist)
            print(reverse(generated, self.data.vocab))
            input()

    # TODO: evaluate and write to file!
    def evaluate(self):
        # eval()
        # no_grad()
        raise NotImplementedError


class SupervisedTrainer(BaseTrainer):
    def __init__(self, model, data, optimizer, scheduler, clip, stats, savedir,
                 backward):
        super().__init__(model, data, optimizer, scheduler, clip, stats, savedir)
        self.criterion = nn.NLLLoss(ignore_index=PAD_IDX)
        self.backward = backward
        self.config()

    # DEBUG: error when 'model instantiation - train' sequence gets mixed up in main.py
    def config(self):
        if self.backward: # to use packed_sequence
            sort_key = lambda ex: (len(ex.resp), len(ex.hist2))
        else:
            sort_key=lambda ex: (len(ex.merged_hist), len(ex.resp))
        self.data.train_iter.sort_key = sort_key
        self.data.valid_iter.sort_key = sort_key
        self.data.test_iter.sort_key = sort_key

    def _compute_loss(self, batch):
        if self.backward:
            logprobs, attn_weights = self.model(truncate(batch.resp, 'sos'),
                                                truncate(batch.hist2, 'eos'))
            target, _ = truncate(batch.hist2, 'sos')
        else:
            logprobs, attn_weights = self.model(truncate(batch.merged_hist, 'sos'),
                                                truncate(batch.resp, 'eos'))
            target, _ = truncate(batch.resp, 'sos')
        B, L, _ = logprobs.size()
        loss = self.criterion(logprobs.contiguous().view(B*L, -1),
                              target.contiguous().view(-1))
        return loss

    def _run_epoch(self, verbose=True):
        self.stats.epoch += 1
        for step, batch in enumerate(self.data.train_iter):
            loss = self._compute_loss(batch)
            self.stats.record_stats(loss)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()

            # report train stats on a regular basis
            if verbose and (step % 100 == 0):
                self.stats.report_stats(step=step)
        return loss.item()


class RLTrainer(BaseTrainer):
    def __init__(self, model, data, optimizer, scheduler, clip, stats, savedir,
                 reward_func, turn,patience=3, metric='Bleu_1'):
        super().__init__(model, data, optimizer, scheduler, clip, stats, savedir)
        # TODO: variable name - criterion?
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX,
                                             reduction='none') # helper
        self.early_stopper = EarlyStopper(patience, metric)
        self.evaluator = NLGEval(no_skipthoughts=True, no_glove=True)
        #self.simulator = Simulator(model, reward_func, turn)

    def _compute_loss(self, batch):
        # calculate loss with simulator
        # ex)
        #    rewards = simulator.simulate(batch)
        #    loss = rewards * self.criterion(a, b)
        return

    def _run_epoch(self):
        # compute loss for every step
        # ex)
        #   for batch in iter:
        #       loss = self._compute_loss(batch)
        #       loss.backward()
        pass

def build_trainer(kind, model, data, lr, lr_shrink, scheduler_patience, clip,
                  records, savedir='models/', backward=False):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_shrink,
                                                     patience=scheduler_patience)
    stats = Stats(records)
    if kind == 'supervised':
        return SupervisedTrainer(model, data, optimizer, scheduler, clip, stats,
                                 savedir, backward)
    elif kind == 'RL':
        pass

