import constants
import cPickle as pickle
import os

class Index(object):
    def __init__(self):
        self._ent_index = dict()
        self._rel_index = dict()

    def rel_to_ind(self, rel):
        if rel not in self._rel_index:
            print("Warning: Key {} not in index".format(rel))
            self._rel_index[rel] = len(self._rel_index.keys())
        return self._rel_index[rel]

    def ent_to_ind(self, ent):
        if ent not in self._ent_index:
            print("Warning: Key {} not in index".format(ent))
            self._ent_index[ent] = len(self._ent_index.keys())
        return self._ent_index[ent]

    def load_index(self,dir_name):
        if os.path.exists(os.path.join(dir_name,constants.entity_ind)):
            self._ent_index = pickle.load(open(os.path.join(dir_name,constants.entity_ind)))
            self._rel_index = pickle.load(open(os.path.join(dir_name, constants.rel_ind)))
        else:
            print("Index not found, creating one.")

    def save_index(self,dir_name):
        pickle.dump(self._ent_index,open(os.path.join(dir_name,constants.entity_ind),'w'))
        pickle.dump(self._rel_index,open(os.path.join(dir_name, constants.rel_ind), 'w'))

    def ent_vocab_size(self):
        return len(self._ent_index)

    def rel_vocab_size(self):
        return len(self._rel_index)

class Path(object):
    def __init__(self, s, r, t):
        assert isinstance(s, int) and isinstance(t, int)
        assert isinstance(r, int)
        self.s = s # source
        self.t = t # target
        self.r = r # relation
        self.pairs = [s,t]
    def __repr__(self):
        rep = "{} {} {}".format(self.s,self.r,self.t)
        return rep

    def __eq__(self, other):
        if not isinstance(other,Path):
            return False
        equal = self.s == other.s and self.t == other.t and self.r == other.r
        return equal

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        hash_p = self.s.__hash__() + self.r.__hash__() + self.t.__hash__()
        return hash_p


def read_dataset(path, results_dir,dev_mode=True,max_examples = float('inf')):
    index = Index()
    index.load_index(results_dir)
    data_set = {}
    data_set['train'] = read_file(os.path.join(path,'train'),index,max_examples)
    if dev_mode:
        data_set['test'] = read_file(os.path.join(path,'dev'),index,max_examples)
    else:
        data_set['test'] = read_file(os.path.join(path, 'test'),index, max_examples)
    data_set['dev'] = read_file(os.path.join(path, 'dev'), index, max_examples)
    data_set['num_ents'] = index.ent_vocab_size()
    data_set['num_rels'] = index.rel_vocab_size()
    index.save_index(results_dir)
    return data_set

def read_file(f_name, index, max_examples):
    data = []
    count = 0
    with open(f_name) as f:
        for line in f:
            if count >= max_examples:
                return data
            s,r,t = line.strip().split("\t")
            p = Path(index.ent_to_ind(s), index.rel_to_ind(r), index.ent_to_ind(t))
            data.append(p)
            count += 1
    return data