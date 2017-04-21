import util
import numpy as np
import time
import os
from data_loader import Path
from sys import stdout
import constants

class Evaluator(object):
    def __init__(self,model,neg_sampler):
        self.ns = neg_sampler
        self.model = model

    def evaluate(self,batch):
        raise NotImplementedError()


class RankEvaluator(Evaluator):
    def __init__(self,model,neg_sampler):
        super(RankEvaluator,self).__init__(model,neg_sampler)
        self.init_score = float('inf') # because lower the better
        self.metric_name = "Mean Rank"
        self.tol = 0.1

    def comparator(self,curr_score,prev_score):
        # write if curr_score less than prev_score
        return curr_score < prev_score + self.tol

    def evaluate(self,batch):
        s_negs = self.ns.batch_sample(batch, False)
        t_negs = self.ns.batch_sample(batch, True)
        s_scores = self.model.predict(batch,s_negs, False).data.cpu().numpy()
        t_scores = self.model.predict(batch, t_negs,True).data.cpu().numpy()
        s_ranks = util.ranks(s_scores, ascending=False)
        t_ranks = util.ranks(t_scores, ascending=False)
        return (np.mean(s_ranks) + np.mean(t_ranks))/2.


class TestEvaluator(Evaluator):
    def __init__(self,model,neg_sampler,results_dir):
        super(TestEvaluator,self).__init__(model,neg_sampler)
        self.all_ranks = []
        self.results_dir =results_dir


    def evaluate(self,batch):
        rep_steps = 10
        rr,hits_10 = (0.0,0.0)
        pos = self.model.predict(batch,None).cpu().data.numpy()
        ind = 0
        start = time.time()
        for ex, p in zip(batch, pos):
            s_rank,t_rank = self.compute_metrics(p,ex)
            rr,hits_10 = self.metrics(s_rank,rr,hits_10,ind)
            ind += 1
            rr, hits_10 = self.metrics(t_rank, rr, hits_10, ind)
            ind += 1
            if ind % (rep_steps*2) == 0:
                end = time.time()
                secs = (end - start)
                stdout.write("\rSpeed {} qps. Percentage complete {}, MRR {}, HITS@10 {} ".
                             format(rep_steps / float(secs), 0.5*ind/float(len(batch)) * 100,rr,hits_10))
                stdout.flush()
                start = time.time()
        self.write_ranks()
        return rr, hits_10

    def write_ranks(self):
        all_ranks = [str(x) for x in self.all_ranks]
        with open(os.path.join(self.results_dir, 'ranks_checkpoint'), 'w') as f:
            f.write("\n".join(all_ranks))

    def metrics(self,rank,rr,hits_10,ind):
        rr = (rr * ind + 1.0 / float(rank)) / float(ind + 1)
        h_10 = 1.0 if rank <= 10 else 0.0
        hits_10 = (hits_10 * ind + h_10) / float(ind + 1)
        self.all_ranks.append(rank)
        return rr,hits_10


    def compute_metrics(self,pos,ex):

        def calc_scores(batches):
            scores = []
            for b in batches:
                scores.extend(self.model.predict(b,None).cpu().data.numpy().tolist())
            scores.append(pos)
            assert pos == scores[-1]
            scores = np.asarray(scores)
            ranks = util.ranks(scores.ravel(), ascending=False)
            return ranks[-1]

        s_negs= self.ns.bordes_negs(ex,False)
        t_negs = self.ns.bordes_negs(ex,True)
        s_negs  = self.pack_negs(ex, s_negs,False)
        t_negs  = self.pack_negs(ex, t_negs, True)
        negs_s = util.chunk(s_negs,constants.test_batch_size)
        negs_t = util.chunk(t_negs, constants.test_batch_size)
        s_rank = calc_scores(negs_s)
        t_rank = calc_scores(negs_t)
        return s_rank,t_rank

    def pack_negs(self,ex,negs,is_target):
        batch = []
        for n in negs:
            p = Path(ex.s,ex.r,n) if is_target else Path(n,ex.r,ex.t)
            batch.append(p)

        return batch