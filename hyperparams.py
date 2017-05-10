import numpy as np
import json
import constants

def main():
    data = 'freebase'
    base = "/home/mitarb/kotnis/Data/neg_sampling/"
    #models = {'rescal','transE','distmult','complex'}
    models = {'distmult'}
    l2 = 0.00024036 # from hyper-param tuning
    for model in models:
        #num_negs(model,data,base,l2)
        tune_l2(model,data,base)

def num_negs(model,data,base,l2):
    path = base + "{}/experiment_specs/".format(data)
    exp_name = "{}".format(model) + "{}.json"
    config = create_config(model,l2=l2)
    negs = [1,2,5,20,50,100]
    for n in negs:
        config['num_negs'] = n
        json.dump(config, open(path + exp_name.format("_" + str(n)), 'w'),
                  sort_keys=True, separators=(',\n', ':'))

def tune_l2(model,data,base):
    path = base+"{}/experiment_specs/".format(data)
    exp_name = "{}".format(model) + "{}.json"
    config = create_config(model)
    l2 = np.sort(np.random.uniform(2,4,size=4))

    for count,e in enumerate(l2):
            config['l2'] = np.power(10,-e)
            json.dump(config,open(path+exp_name.format("_"+str(count+1)),'w'),
                      sort_keys=True,separators=(',\n', ':'))

def create_config(model_name,neg_sampler='random',l2=0):
    config = dict()
    config['model'] = model_name
    config['lr'] = 0.001
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