import numpy as np
from torch.autograd import Variable
import torch
from scipy import stats
cache = dict()

def chunk(arr,chunk_size):
    if len(arr)==0:
        yield []

    for i in range(0,len(arr),chunk_size):
        yield arr[i:i+chunk_size]


def sample(data,num_samples,replace=False):

    if len(data) <= num_samples:
        return data
    idx = np.random.choice(len(data),num_samples,replace=replace)
    return [data[i] for i in idx]


def ranks(scores, ascending = False):
    sign = 1 if ascending else -1
    scores = scores * sign
    ranks = [stats.rankdata(scores[i])[0] for i in range(scores.shape[0])]
    return np.mean(ranks)

def get_triples(batch,negs,is_target=True, volatile=False):
    sources,rels,targets = ([],[],[])
    if negs is None:
        for ex in batch:
            sources.append([ex.s])
            targets.append([ex.t])
            rels.append(ex.r)
    else:
        for count,ex in enumerate(batch):
            s = [] if is_target else [n for n in negs[count]]
            t = [n for n in negs[count]] if is_target else []
            s.insert(0,ex.s)
            t.insert(0,ex.t)
            sources.append(s)
            targets.append(t)
            rels.append(ex.r)

    return to_var(sources,volatile=volatile), to_var(targets,volatile=volatile),to_var(rels,volatile=volatile)

def to_var(x,volatile=False):
    if 'cuda' not in cache:
        cache['cuda'] = torch.cuda.is_available()
    cuda = cache['cuda']
    var = Variable(torch.from_numpy(np.asarray(x)),volatile=volatile)
    if cuda:
        return var.cuda()
    return var