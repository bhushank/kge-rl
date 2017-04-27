import time
import util
import negative_sampling
import data_loader
import multiprocessing as mp
data_path = '/home/mitarb/kotnis/Data/neg_sampling/freebase/'
results_dir = data_path + 'rescal_1/'
def sample(ns,ex):
    return ns.sample(ex,True)

data = data_loader.read_dataset(data_path,results_dir,dev_mode=True,max_examples=float('inf'))
ns  = negative_sampling.Random_Sampler(data['train'],100,filtered=False)
batch = data['train'][:4000]

start = time.time()
for ex in batch:
    samples = sample(ns,ex)
end = time.time()
print("Single Process {}".format(end-start))