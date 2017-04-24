import copy
import numpy as np


class Negative_Sampler(object):
    def __init__(self,triples,num_samples,filtered):
        self.num_samples = num_samples
        self._triples = set(triples)
        self._entities = self._get_entities(triples)
        self._entity_set = set(copy.copy(self._entities))
        self.filtered = filtered
        self.s_filter = self._compute_filter(False)
        self.t_filter = self._compute_filter(True)

    def batch_sample(self, batch, is_target, num_samples=0):
        # Single Threaded #ToDO: Parallelize using multiproc
        batched_negs = [self.sample(ex, is_target, num_samples) for ex in batch]
        return batched_negs

    def sample(self,ex,is_target,num_samples):
        raise NotImplementedError("Abstract method")

    def filter_candidates(self,ex,is_target,candidates):
        if self.filtered:
            known_candidates = self.t_filter[(ex.s, ex.r)] if is_target else self.s_filter[(ex.r, ex.t)]
            for e in known_candidates:
                candidates.remove(e)
        gold = ex.t if is_target else ex.s
        if gold in candidates:
            candidates.remove(gold)
        return candidates

    def _get_entities(self,data):
        entities  = set()
        for ex in data:
            entities.add(ex.s)
            entities.add(ex.t)
        return list(entities)

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

class Random_Sampler(Negative_Sampler):
    def __init__(self,triples,num_samples,filtered=False):
        super(Random_Sampler,self).__init__(triples,num_samples,filtered)

        print("Neg. Sampler: Random, num_samples: {}, filtered: {}".format(num_samples,filtered))

    def sample(self,ex,is_target,num_samples=0):
        candidates = self._entity_set.copy()
        samples = self.filter_candidates(ex,is_target,candidates)

        num_samples = self.num_samples if num_samples < self.num_samples else num_samples
        if num_samples==float('inf'):
            assert self.filtered or len(samples)==14950
            return list(samples)

        samples = np.random.choice(list(samples), num_samples, replace=False).tolist()
        return samples


class Corrupt_Sampler(Negative_Sampler):
    def __init__(self,triples,num_samples,filtered=False):
        super(Corrupt_Sampler,self).__init__(triples,num_samples,filtered)
        self._typed_entities = self.get_typed()
        print("Neg. Sampler: Corrupt, num_samples: {}, filtered: {}".format(num_samples, filtered))
        self.rs = Random_Sampler(triples,num_samples)

    def get_typed(self):
        typed = dict()
        for ex in self._triples:
            ents = typed.get(ex.r,tuple([set(),set()]))
            ents[0].add(ex.s)
            ents[1].add(ex.t)
            typed[ex.r] = ents
        return typed


    def pad_samples(self,ex,samples, n,is_target):
        while True:
            if len(samples) == n:
                break
            new_samples = self.rs.sample(ex, n - len(samples), is_target)
            samples.update(set(new_samples))
        return list(samples)

    def sample(self,ex,is_target,num_samples=0):
        candidates = self._typed_entities[ex.r][1] if is_target else self._typed_entities[ex.r][0]
        samples = self.filter_candidates(ex,is_target,candidates)
        # if corrupted negatives less than num_samples then augment with random samples
        if num_samples >= len(samples):
            return self.pad_samples(ex,samples,num_samples,is_target)
        samples = np.random.choice(list(samples), num_samples, replace=False).tolist()
        return samples