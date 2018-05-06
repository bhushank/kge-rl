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
from embedding_loader import save_embeddings

def main(exp_name,data_path,resume,tune,vectors):
    torch.manual_seed(32345)
    print("Pytorch Version {}".format(torch.__version__))
    
    if vectors:
        cuda = torch.cuda.is_available()
        config = json.load(open(os.path.join(data_path, "{}".format(exp_name), "config.json".format(exp_name))))
        print("Saving Embeddings")
        save_embeddings(os.path.join(data_path,exp_name),config['model'],is_cpu=not cuda)
        print("Embeddings Saved.")
        exit(0)
    config = json.load(open(os.path.join(data_path, 'experiment_specs', "{}.json".format(exp_name))))
    operation = config.get('operation','train_test')
    if operation=='train':
        train(config,exp_name,data_path,resume,tune)
    elif operation=='test':
        test(config,exp_name,data_path)
    elif operation=='train_test':
        train_test(config,exp_name,data_path,resume,tune)
    else:
        raise NotImplementedError("{} Operation Not Implemented".format(operation))



def train_test(config,exp_name,data_path,resume=False,tune=False):
    train(config,exp_name,data_path,resume,tune)
    test(config,exp_name,data_path)


def train(config,exp_name,data_path,resume=False,tune=False):

    results_dir =  os.path.join(data_path,exp_name)
    if os.path.exists(results_dir) and not (resume or tune):
        print("{} already exists, no need to train.\n".format(results_dir))
        return
    if not os.path.exists(results_dir):
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
    # Provide train and dev data to negative sampler for filtering positives
    data = copy.copy(data_set['train'])
    data.extend(data_set['test'])
    model,neg_sampler,evaluator = build_model(data,config,
                                              results_dir,data_set['num_ents'],data_set['num_rels'])
    model = is_gpu(model,cuda)
    state = None
    if resume or tune:
        params_path = os.path.join(results_dir, '{}_params.pt'.format(config['model']))
        model.load_state_dict(torch.load(params_path))
    if resume:
        state_path = os.path.join(results_dir,'{}_optim_state.pt'.format(config['model']))
        state = torch.load(state_path)
    if config['neg_sampler'] == 'rl':
        sgd = optimizer.Reinforce(data_set['train'],data_set['dev'],model,
                        neg_sampler,evaluator,results_dir,config,state)
    else:
        sgd = optimizer.SGD(data_set['train'],data_set['dev'],model,
                        neg_sampler,evaluator,results_dir,config,state)

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
    state_dict = torch.load(params_path) if cuda \
        else torch.load(params_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()
    print("Filtered Setting")
    evaluate(data_set['test'],evaluator,results_dir,is_dev,True)
    #if not is_dev:
        #evaluator.ns.filtered=False
        #print("Raw Setting, filtered: {}".format(evaluator.ns.filtered))
        #evaluate(data_set['test'], evaluator, results_dir, is_dev, False)


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
            qc = "Query Count : {}".format((count + 1)*constants.test_batch_size)
            metrics ="Mean Reciprocal Rank : {:.4f}, HITS@10 : {:.4f}".format(mrr,h10)
            print ("{}, {}, {}".format(speed,qc,metrics))
            start = time.time()

    print('Writing Results.')
    split = 'dev' if is_dev else 'test'
    filt = 'filt' if filtered else 'raw'
    all_ranks = [str(x) for x in evaluater.all_ranks]
    with open(os.path.join(results_dir,'ranks_{}_{}'.format(split,filt)),'w') as f:
        f.write("\n".join(all_ranks) +"\n")
    with open(os.path.join(results_dir,'results_{}_{}'.format(split,filt)),'w') as f:
        f.write("Mean Reciprocal Rank : {:.4f}\nHITS@10 : {:.4f}\n".
                format(mrr,h10))

def build_model(triples,config,results_dir,n_ents,n_rels,train=True,filtered=True):

    def get_model():
        if config['model']=='rescal':
            return models.Rescal(n_ents,n_rels,config['ent_dim'])
        elif config['model']=='transE':
            return models.TransE(n_ents, n_rels, config['ent_dim'])
        elif config['model']=='distmult':
            return models.Distmult(n_ents, n_rels, config['ent_dim'])
        elif config['model']=='complex':
            return models.ComplEx(n_ents, n_rels, config['ent_dim'])
        else:
            raise NotImplementedError("Model {} not implemented".format(config['model']))

    def  get_neg_sampler(model=None):
        if not train:
            return negative_sampling.Random_Sampler(triples,float('inf'),filtered=filtered)
        elif config['neg_sampler'] == 'random':
            return negative_sampling.Random_Sampler(triples,config['num_negs'])
        elif config['neg_sampler'] == 'corrupt':
            return negative_sampling.Corrupt_Sampler(triples,config['num_negs'])
        elif config['neg_sampler'] == 'typed':
            return negative_sampling.Typed_Sampler(triples,config['num_negs'],results_dir)
        elif config['neg_sampler'] == 'relational':
            return negative_sampling.Relational_Sampler(triples,config['num_negs'])
        elif config['neg_sampler'] == 'nn':
            return negative_sampling.NN_Sampler(triples,config['num_negs'])
        elif config['neg_sampler'] == 'adversarial':
            return negative_sampling.Adversarial_Sampler(triples, config['num_negs'])
        elif config['neg_sampler'] == 'rl':
            return negative_sampling.Policy_Sampler(triples, config['num_negs'])
        else:
            raise NotImplementedError("Neg. Sampler {} not implemented".format(config['neg_sampler']))

    model = get_model()
    ns = get_neg_sampler(model)
    if train:
        print('Evaluation Sampler')
        eval_ns = negative_sampling.Random_Sampler(triples,constants.num_dev_negs)
        evaluator = RankEvaluator(model,eval_ns)
    else:
        test_ns = negative_sampling.Test_Sampler(triples,constants.num_dev_negs)
        evaluator = TestEvaluator(model,test_ns,results_dir)
    return model,ns,evaluator

def is_gpu(model,cuda):
    if cuda:
        model.cuda()
        print("Using GPU {}".format(torch.cuda.current_device()))
    else:
        print("Using CPU")
        #torch.set_num_threads(40)
    return model

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('exp_name')
    parser.add_argument('-r', action='store_true')
    parser.add_argument('-t', action='store_true')
    parser.add_argument('-v', action='store_true')
    args = parser.parse_args()
    main(args.exp_name,args.data_path,args.r,args.t,args.v)

