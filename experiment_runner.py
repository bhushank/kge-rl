import argparse
import json
import os
from evaluation import TestEvaluator, RankEvaluator
import models
import optimizer
import util
import time
import negative_sampling
import constants
import copy
import torch
import data_loader

def main(exp_name,data_path,resume):
    config = json.load(open(os.path.join(data_path,'experiment_specs',"{}.json".format(exp_name))))
    print("Pytorch Version {}".format(torch.__version__))
    operation = config.get('operation','train_test')
    if operation=='train':
        train(config,exp_name,data_path,resume)
    elif operation=='test':
        test(config,exp_name,data_path)
    elif operation=='train_test':
        train_test(config,exp_name,data_path,resume)
    else:
        raise NotImplementedError("{} Operation Not Implemented".format(operation))

def train_test(config,exp_name,data_path,resume=False):
    train(config,exp_name,data_path,resume)
    test(config,exp_name,data_path)


def train(config,exp_name,data_path,resume=False):

    results_dir =  os.path.join(data_path,exp_name)
    if os.path.exists(results_dir):
        print("{} already exists, no need to train.\n".format(results_dir))
        return
    os.makedirs(results_dir)
    json.dump(config,open(os.path.join(results_dir,'config.json'),'w'),
              sort_keys=True,separators=(',\n', ': '))
    is_dev = config['is_dev']
    print("\n***{} MODE***\n".format('DEV' if is_dev else 'TEST'))
    if not is_dev:
        print("\n***Changing TEST to DEV***\n")
        config['is_dev'] = True
    data_set = data_loader.read_dataset(data_path,results_dir,dev_mode=True,max_examples=float('inf'))
    cuda = torch.cuda.is_available()

    print("Number of training data points {}".format(len(data_set['train'])))
    print("Number of dev data points {}".format(len(data_set['test'])))

    model,neg_sampler,evaluator = build_model(data_set['train'],config,
                                              results_dir,data_set['num_ents'],data_set['num_rels'])
    model = is_gpu(model,cuda)
    if resume:
        params_path = os.path.join(results_dir, '{}_params.pt'.format(config['model']))
        model.load_state_dict(torch.load(params_path))

    sgd = optimizer.SGD(data_set['train'],data_set['dev'],model,
                        neg_sampler,evaluator,results_dir,config)

    start = time.time()
    sgd.minimize()
    end = time.time()
    hours = int((end-start)/ 3600)
    minutes = ((end-start) % 3600) / 60.
    profile_string = "Finished Training! Took {} hours and {} minutes\n".format(hours,minutes)
    with open(os.path.join(results_dir,'train_time'),'w') as f:
        f.write(profile_string+"Raw seconds {}\n".format(end-start))
    print(profile_string)



def test(config,exp_name,data_path):

    print("Testing...\n")
    is_dev = config['is_dev']
    cuda =  torch.cuda.is_available()
    print("\n***{} MODE***\n".format('DEV' if is_dev else 'TEST'))
    results_dir = os.path.join(data_path, exp_name)
    params_path = os.path.join(results_dir,'{}_params.pt'.format(config['model']))
    if not os.path.exists(params_path):
        print("No trained params found, quitting.")
        return

    data_set = data_loader.read_dataset(data_path,results_dir,dev_mode=is_dev)
    all_data = copy.copy(data_set['train'])
    all_data.extend(data_set['dev'])
    if not is_dev:
        all_data.extend(data_set['test'])

    model,neg_sampler,evaluator = build_model(all_data,config,results_dir,
                                              data_set['num_ents'],data_set['num_rels'],train=False)
    model = is_gpu(model, cuda)
    model.load_state_dict(torch.load(params_path))
    model.eval()
    print("Filtered Setting")
    evaluate(data_set['test'],evaluator,results_dir,is_dev,True)
    if not is_dev:
        print("Raw Setting")
        evaluate(data_set['test'], evaluator, results_dir, is_dev, False)


def evaluate(data,evaluater,results_dir,is_dev,filtered):
    print("Evaluating")
    h10,mrr = 0.0,0.0
    start = time.time()
    report_period = 1
    for count,d in enumerate(util.chunk(data,constants.test_batch_size)):
        rr, hits_10 = evaluater.evaluate(d)
        h10 = (h10*count + hits_10)/float(count + 1)
        mrr = (mrr*count + rr)/float(count+1)
        if count%report_period==0:
            end = time.time()
            secs = (end - start)
            speed = "Speed {} queries per second".format(report_period*constants.test_batch_size/float(secs))
            qc = "Query Count : {}".format(count)
            metrics ="Mean Reciprocal Rank : {:.4f}, HITS@10 : {:.4f}".format(mrr,h10)
            print ("{}, {}, {}".format(speed,qc,metrics))
            start = time.time()

    print('Writing Results.')
    split = 'dev' if is_dev else 'test'
    filt = 'filt' if filtered else 'raw'
    all_ranks = [str(x) for x in evaluater.all_ranks]
    with open(os.path.join(results_dir,'ranks_{}_{}'.format(split,filt)),'w') as f:
        f.write("\n".join(all_ranks))
    with open(os.path.join(results_dir,'results_{}_{}'.format(split,filt)),'w') as f:
        f.write("Mean Reciprocal Rank : {:.4f}\nHITS@10 : {:.4f}\n".
                format(mrr,h10))


def build_model(triples,config,results_dir,n_ents,n_rels,train=True):

    def get_model():
        if config['model']=='rescal':
            return models.Rescal(n_ents,n_rels,config['ent_dim'])
        elif config['model']=='transE':
            return models.TransE(n_ents, n_rels, config['ent_dim'])
        elif config['model']=='distmult':
            return models.TransE(n_ents, n_rels, config['ent_dim'])
        elif config['model']=='complex':
            return models.TransE(n_ents, n_rels, config['ent_dim'])
        else:
            raise NotImplementedError("Model {} not implemented".format(config['model']))

    def  get_neg_sampler():
        if not train:
            return negative_sampling.Random_Sampler(triples,float('inf'),filtered=True)
        if config['neg_sampler'] == 'random':
            return negative_sampling.Random_Sampler(triples,config['num_negs'])
        else:
            raise NotImplementedError("Neg. Sampler {} not implemented".format(config['neg_sampler']))

    model = get_model()
    ns = get_neg_sampler()
    evaluator = RankEvaluator(model,ns) if train \
        else TestEvaluator(model,ns,results_dir)
    return model,ns,evaluator

def is_gpu(model,cuda):
    if cuda:
        model.cuda()
        print("Using GPU {}".format(torch.cuda.current_device()))
    else:
        print("Using CPU")
        torch.set_num_threads(56)
    return model

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('exp_name')
    parser.add_argument('-r', action='store_true')
    args = parser.parse_args()
    main(args.exp_name,args.data_path,args.r)

