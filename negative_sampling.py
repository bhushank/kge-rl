import copy
import os
import constants
from sample_list import sample_list
import cPickle as pickle
from sklearn.neighbors import BallTree

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
        batched_negs = [self.sample(ex, is_target, num_samples) for ex in batch]
        return batched_negs

    def sample(self,ex,is_target,num_samples):
        raise NotImplementedError("Abstract method")

    def filter_candidates(self,ex,is_target,candidates):
        if self.filtered:
            known_candidates = self.t_filter[(ex.s, ex.r)] if is_target else self.s_filter[(ex.r, ex.t)]
            candidates = candidates.difference(known_candidates)
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
        num_samples = self.num_samples if num_samples <= 0 else num_samples
        if num_samples==float('inf'):
            assert self.filtered or len(samples)==14950
            return list(samples)
        #samples = np.random.choice(list(samples), num_samples, replace=False).tolist()
        samples = sample_list(list(samples),num_samples)
        assert len(samples) >= 1
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
        assert len(typed.keys())==constants.fb15k_rels
        return typed


    def pad_samples(self,ex,samples, n,is_target):
        while True:
            if len(samples) == n:
                break
            new_samples = self.rs.sample(ex, is_target,n - len(samples))
            samples.update(set(new_samples))
        return list(samples)

    def sample(self,ex,is_target,num_samples=0):
        candidates = self._typed_entities[ex.r][1] if is_target else self._typed_entities[ex.r][0]
        samples = self.filter_candidates(ex,is_target,candidates)
        num_samples = self.num_samples if num_samples <= 0 else num_samples
        # if corrupted negatives less than num_samples then augment with random samples
        if num_samples >= len(samples):
            return self.pad_samples(ex,samples,num_samples,is_target)
        #samples = np.random.choice(list(samples), num_samples, replace=False).tolist()
        samples = sample_list(list(samples),num_samples)
        assert len(samples) >= 1
        return samples

class Typed_Sampler(Negative_Sampler):
    def __init__(self,triples,num_samples,results_dir,filtered=True):
        super(Typed_Sampler, self).__init__(triples,num_samples,filtered)
        print("Neg. Sampler: Typed, num_samples: {}, filtered: {}".format(num_samples, filtered))
        self.ent_index = pickle.load(open(os.path.join(results_dir, constants.entity_ind)))
        self.ent_cats,self.cat_ents = self.load_cats()
        self.rs = Random_Sampler(triples, num_samples)

    def load_cats(self):
        ent_cats, cat_ents = dict(),dict()
        all_cats = pickle.load(open(constants.cat_file))
        for k,v in all_cats.iteritems():
            ent_cats[self.ent_index[k]] = v
            for c in v:
                ents = cat_ents.get(c, set())
                ents.add(self.ent_index[k])
                cat_ents[c] = ents
        return ent_cats,cat_ents

    def pad_samples(self, ex, samples, n, is_target):
        while True:
            if len(samples) == n:
                break
            new_samples = self.rs.sample(ex, is_target, n - len(samples))
            samples.update(set(new_samples))
        return list(samples)

    def sample(self, ex, is_target, num_samples=0):
        def get_candidates():
            candidates = set()
            entity = ex.t if is_target else ex.s
            cats = self.ent_cats.get(entity,set())
            for c in cats:
                candidates.update(self.cat_ents.get(c,set()))
            return candidates

        candidates = get_candidates()
        samples = self.filter_candidates(ex, is_target, candidates)
        num_samples = self.num_samples if num_samples <= 0 else num_samples
        # if corrupted negatives less than num_samples then augment with random samples
        if num_samples >= len(samples):
            return self.pad_samples(ex, samples, num_samples, is_target)
        #samples = np.random.choice(list(samples), num_samples, replace=False).tolist()
        samples = sample_list(list(samples), num_samples)
        assert len(samples) >= 1
        return samples


class Dynamic_Sampler(Negative_Sampler):
    def __init__(self, triples, num_samples, model, filtered=True):
        super(Dynamic_Sampler, self).__init__(triples, num_samples, filtered)
        self.model = model
        self.create_ball_tree()

    def batch_sample(self, batch, is_target, num_samples=0):
        raise NotImplementedError("Abstract Method")

    def create_ball_tree(self):
        entities = self.model.all_entity_vectors()
        self.tree = BallTree(entities,leaf_size=entities.shape[0])

    def sample(self,ex,is_target,entity_vector,num_samples=0):
        num_samples = self.num_samples if num_samples <= 0 else num_samples
        ind = self.tree.query([entity_vector], k=num_samples,return_distance=False)
        samples = self.filter_candidates(ex,is_target,set(ind[0]))
        curr_num = num_samples
        while True:
            if len(samples)==num_samples:
                return list(samples)
            diff = num_samples - len(samples)
            ind = self.tree.query([entity_vector], k=curr_num+diff,return_distance=False)
            ind = ind[0,curr_num:]
            new_samples  = self.filter_candidates(ex,is_target,set(ind))
            samples.update(new_samples)
            curr_num += diff

    def get_entity(self,ex,is_target):
        return ex.t if is_target else ex.s

class NN_Sampler(Dynamic_Sampler):
    def __init__(self,triples,num_samples,model,filtered=True):
        super(NN_Sampler,self).__init__(triples,num_samples,model,filtered)
        print("Neg. Sampler: Nearest Neighbors, num_samples: {}, filtered: {}"
              .format(num_samples, filtered))



    def batch_sample(self, batch, is_target, num_samples=0):

        entities = [self.get_entity(ex,is_target) for ex in batch]
        entity_vectors = self.model.entity_vectors(entities)
        batched_negs = [self.sample(ex, is_target,e_v,num_samples) for ex,e_v in zip(batch,entity_vectors)]
        return batched_negs


class Adversarial_Sampler(Dynamic_Sampler):
    def __init__(self,triples,num_samples,model,filtered=True):
        super(Adversarial_Sampler,self).__init__(triples,num_samples,model,filtered)
        print("Neg. Sampler: Adversarial, num_samples: {}, filtered: {}"
              .format(num_samples, filtered))

    def batch_sample(self, batch, is_target, num_samples=0):
        entities = []
        rels = []
        for ex in batch:
            rels.append(ex.r)
            # if is_target (predicting target), then should use sources
            entities.append(self.get_entity(ex,not is_target))
        entity_vectors = self.model.output(entities,rels,is_target)
        batched_negs = [self.sample(ex, is_target, e_v, num_samples) for ex, e_v in zip(batch, entity_vectors)]
        return batched_negs

