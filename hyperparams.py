import numpy as np
import json
import constants

def main():
    data = 'freebase'
    base = "/home/kotnis/data/neg_sampling/"
    models = {'rescal','transE','distmult','complex'}
    for model in models:
        create_json(model,data,base)

def create_json(model,data,base):
    path = base+"{}/experiment_specs/".format(data)
    exp_name = "{}".format(model) + "{}.json"
    config = create_config(model)
    l2 = np.sort(np.random.uniform(1,4,size=4))
    count = 1
    for e in l2:
            config['l2'] = np.power(10,-e)
            json.dump(config,open(path+exp_name.format("_"+str(count)),'w'),
                      sort_keys=True,separators=(',\n', ':'))
            count+=1


def create_config(model_name):
    config = dict()
    config['model'] = model_name
    config['lr'] = 0.001
    config['batch_size'] = constants.batch_size
    config['neg_sampler'] = 'random'
    config['num_negs'] = 10
    config['num_epochs']= 100
    config['is_dev'] = True
    config['ent_dim'] = 100
    return config

if __name__=='__main__':
    main()