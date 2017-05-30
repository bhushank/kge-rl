import numpy as np
import json
import constants
import os
def main():
    data = 'wordnet'
    base = "/home/kotnis/data/neg_sampling/"
    models = {'rescal':7.484410236920948e-05,'transE':0.0001863777691779108,'distmult':3.120071843121878e-06,'complex': 2.8198448631731174e-05}
    samplers = {"random","corrupt","relational","nn","adversarial"}
    #models = {'complex'}
    l2 = 1.3074905074564395e-06# from hyper-param tuning
    for model,l2 in models.iteritems():
        for sampler in samplers:
            num_negs(model,data,base,l2,sampler)
        #tune_l2(model,data,base)

def num_negs(model,data,base,l2,sampler):
    if not os.path.exists(base + "{}/experiment_specs/{}".format(data,sampler)):
        os.mkdir(base + "{}/experiment_specs/{}".format(data,sampler))
    path = base + "{}/experiment_specs/{}/".format(data,sampler)
    exp_name = "{}".format(model) + "{}.json"
    config = create_config(model,sampler,l2)
    negs = [1,2,5,10,20,50,100]
    for n in negs:
        config['num_negs'] = n
        json.dump(config, open(path + exp_name.format("_" + str(n)), 'w'),
                  sort_keys=True, separators=(',\n', ':'))

def tune_l2(model,data,base):
    path = base+"{}/experiment_specs/".format(data)
    exp_name = "{}".format(model) + "{}.json"
    config = create_config(model,'random',0.0)
    l2 = np.sort(np.random.uniform(3.5,6,size=4))

    for count,e in enumerate(l2):
            config['l2'] = np.power(10,-e)
            json.dump(config,open(path+exp_name.format("_"+str(count+1)),'w'),
                      sort_keys=True,separators=(',\n', ':'))

def create_config(model_name,neg_sampler,l2):
    config = dict()
    config['model'] = model_name
    config['lr'] = 0.01
    config['l2'] = l2
    config['batch_size'] = constants.batch_size
    config['neg_sampler'] = neg_sampler
    config['num_negs'] = 10
    config['num_epochs']= 100
    config['is_dev'] = False
    config['ent_dim'] = 100
    return config

if __name__=='__main__':
    main()