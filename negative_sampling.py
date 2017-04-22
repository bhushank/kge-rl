import copy
import numpy as np

class Negative_Sampler(object):
    def __init__(self,triples,num_samples):
        self.num_samples = num_samples
        self._triples = set(triples)
        self._entities = self._get_entities(triples)
        self._entity_set = set(copy.copy(self._entities))


    def batch_sample(self,batch,is_target):
        raise NotImplementedError("Abstract method")

    def sample(self,batch,is_target):
        raise NotImplementedError("Abstract method")


    def _get_entities(self,data):
        entities  = set()
        for ex in data:
            entities.add(ex.s)
            entities.add(ex.t)
        return list(entities)

class Random_Sampler(Negative_Sampler):
    def __init__(self,triples,num_samples,filtered=False):
        super(Random_Sampler,self).__init__(triples,num_samples)
        self.filtered = filtered
        self.s_filter = self._compute_filter(False)
        self.t_filter = self._compute_filter(True)
        print("Neg. Sampler: Random, num_samples: {}, filtered: {}".format(num_samples,filtered))

    def sample(self,ex,is_target,num_samples=0):
        samples = self._entity_set.copy()
        if self.filtered:
            known_candidates = self.t_filter[(ex.s, ex.r)] if is_target else self.s_filter[(ex.r, ex.t)]
            for e in known_candidates:
                samples.remove(e)

        gold = ex.t if is_target else ex.s
        if gold in samples:
            samples.remove(gold)
        assert len(samples) > 1
        num_samples = self.num_samples if num_samples < self.num_samples else num_samples
        if num_samples==float('inf'):
            assert len(samples)<=14950
            return list(samples)
        samples = np.random.choice(list(samples), num_samples, replace=False).tolist()
        return samples

    def batch_sample(self,batch,is_target,num_samples=0):
        #Single Threaded #ToDO: Parallelize using multiproc
        batched_negs = [self.sample(ex,is_target) for ex in batch]
        return batched_negs

    def _compute_filter(self,is_target):
        '''
        Returns a dictionary with targets/sources for a (source,rel)/(target,rel) pair.
        This is for the filtered setting
        :param is_target: 
        :return: 
        '''
        filter = dict()
        for ex in self._triples:
            key = (ex.s,ex.r) if is_target else (ex.r,ex.t)
            candidates = filter.get(key,set())
            if is_target:
                candidates.add(ex.t)
            else:
                candidates.add(ex.s)
            filter[key] = candidates
        return filter