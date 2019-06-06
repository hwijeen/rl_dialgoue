import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# TODO: lr_scheduler

def cal_running_avg_loss(loss, running_avg_loss, decay=0.99):
    if running_avg_loss == 0:
        return loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
        return running_avg_loss


class Stats:
    def __init__(self, records):
        self.epoch = 0
        self.records = records
        self.reset_stats()

    def reset_stats(self):
        self.throughout = {name: [] for name in self.records} # for plotting
        self.running_avg = {name: 0 for name in self.records}

    def record_stats(self, *args):
        assert len(self.records) == len(args)
        for name, loss in zip(self.records, args):
            self.throughout[name].append(loss.item())
            self.running_avg[name] = cal_running_avg_loss(loss.item(),
                                                          self.running_avg[name])
    def report_stats(self, step='N/A'):
        logger.info('stats at epoch {} step {}: {}'.format(self.epoch, step,
                                                           str(self.running_avg)))

class EarlyStopper:
    def __init__(self, patience, metric):
        self.patience = patience
        self.metric = metric # 'Bleu_1', ..., 'METEOR', 'ROUGE_L'
        self.count = 0
        self.best_score = defaultdict(lambda: 0)
        self.is_improved = False

    def stop(self, cur_score):
        if self.best_score[self.metric] > cur_score[self.metric]:
            self.is_improved = False
            if self.count <= self.patience:
                self.count += 1
                logger.info('Counting early stop patience... {}' \
                            .format(self.count))
                return False
            else:
                logger.info('Early stopping patience exceeded.\
                            Stopping training...')
                return True # halt training
        else:
            self.is_improved = True
            self.count = 0
            self.best_score = cur_score
            return False
