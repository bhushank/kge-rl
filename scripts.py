import time
import torch
import negative_sampling
import data_loader
import models
''''
data_path = '/home/kotnis/data/neg_sampling/freebase/'
params_path = data_path + 'rescal_params.pt'
results_dir = data_path + 'rescal_1/'
def sample(ns,ex):
    return ns.sample(ex,True)

data = data_loader.read_dataset(data_path,results_dir,dev_mode=True,max_examples=float('inf'))
model = models.Rescal(data['num_ents'], data['num_rels'], 100)
state_dict = torch.load(params_path)
model.load_state_dict(state_dict)
ns  = negative_sampling.NN_Sampler(data['train'],100,model,filtered=False)
batch = data['train'][:4000]
print("Start Profiling")
start = time.time()
samples = ns.batch_sample(batch,True,100)
end = time.time()
print("Time Taken {}".format(end-start))
'''

import numpy as np
n_h = 0
for n in range(500):
    if np.random.uniform() < 0.27:
        n_h+=1

print(n_h/float(n))