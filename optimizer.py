import numpy as np
import util
import time
import constants
import torch
from torch.optim import Adam
import os
from negative_sampling import Dynamic_Sampler, Policy_Sampler
from torch import nn
from torch.autograd import Variable
import operator
import copy

class SGD(object):
    def __init__(self,train,dev,model,negative_sampler,evaluator,results_dir,config,state=None):
        self.train = train
        self.dev = dev
        self.model = model
        self.evaluator = evaluator
        self.ns = negative_sampler
        self.results_dir = results_dir
        self.model_name = config['model']
        #SGD Params
        lr = config.get('lr',0.001)
        l2 = config.get('l2',0.0)
        #self.batch_size = config.get('batch_size',constants.batch_size)
        self.batch_size = config.get('batch_size',constants.batch_size)
        print("lr: {:.4f}, l2: {:.5f}, batch_size: {}".format(lr,l2,self.batch_size))
        self.optim = Adam(model.parameters(),lr=lr,weight_decay=l2)
        if state is not None:
            self.optim.load_state_dict(state)
        #Report and Early Stopping Params
        self.prev_score = evaluator.init_score
        self.early_stop_counter = constants.early_stop_counter
        self.patience = constants.patience
        self.num_epochs = config['num_epochs']

        self.report_steps = constants.report_steps
        self.test_batch_size = config.get('test_batch_size',constants.test_batch_size)
        self.halt = False
        self.dump = True # save without checking

        self.prev_steps = 0
        self.prev_time = time.time()

        #Loss
        self.mm = nn.MarginRankingLoss(margin=1)
        self.bce = torch.nn.BCEWithLogitsLoss()

    def minimize(self):
        print("Training...")
        for epoch in range(self.num_epochs):
            start = time.time()
            train_cp = list(self.train)
            np.random.shuffle(train_cp)
            batches = util.chunk(train_cp, self.batch_size)
            for step,batch in enumerate(batches):
                self.optim.zero_grad()
                loss = self.fprop(batch)
                loss.backward()
                g_norm = torch.nn.utils.clip_grad_norm(self.model.parameters(), 100)
                self.optim.step()
                if step % self.report_steps == 0:
                    self.report(step,g_norm)

            self.prev_steps=0
            self.prev_time=time.time()
            end = time.time()
            mins = int(end - start)/60
            secs = int(end - start)%60
            print("Epoch {} took {} minutes {} seconds".format(epoch+1,mins,secs))
            # Refresh
            self.save(self.dump)
            # Only one epoch for Dynamic Samplers
            if isinstance(self.ns,Dynamic_Sampler) or isinstance(self.ns,Policy_Sampler):
                self.dump = False
                if epoch>=4:
                    self.halt = True
            if self.halt:
                return

    def forward(self,batch,volatile,is_target):
        negs = self.ns.batch_sample(batch, is_target)
        batch = util.get_triples(batch, negs, is_target, volatile=volatile)
        score = self.model(*batch)
        #return self.logistic(score)
        return self.max_margin(score)

    def fprop(self,batch,volatile=False):
        return self.forward(batch,volatile,True) + self.forward(batch, volatile, False)

    def max_margin(self,scores):
        y = util.to_var(np.ones(scores.size()[0],dtype='float32'), requires_grad=False)
        loss = self.mm(scores[:,0],scores[:,1],y)
        for i in range(2,scores.size()[1]):
            loss += self.mm(scores[:,0],scores[:,i],y)
        return loss/(scores.size()[1]-1.)

    def logistic(self,scores):
        y_pos = util.to_var(np.ones(scores.size()[0],dtype='float32'),requires_grad=False)
        y_neg = util.to_var(np.zeros(scores.size()[0], dtype='float32'),requires_grad=False)
        loss = self.bce(scores[:, 0],y_pos)
        for i in range(1,scores.size()[1]):
            loss += self.bce(scores[:, i], y_neg)
        return loss/scores.size()[1]


    def save(self,dump=False):
        curr_score = self.evaluate(self.dev,self.test_batch_size,True)
        print("Current Score: {}, Previous Score: {}".format(curr_score,self.prev_score))
        if self.evaluator.comparator(curr_score, self.prev_score) or dump:
            print("Saving params...\n")
            torch.save(self.model.state_dict(), os.path.join(
                self.results_dir,'{}_params.pt'.format(self.model_name)))
            #Save Optimizer Gradient History for resuming training
            state_path = os.path.join(self.results_dir,"{}_optim_state.pt".format(self.model_name))
            torch.save(self.optim.state_dict(),state_path)
            self.prev_score = curr_score
            # Reset early stop counter
            self.early_stop_counter = self.patience
        else:
            self.early_stop_counter -= 1
            print("New params worse than current, skip saving...\n")

        if self.early_stop_counter <= 0:
            self.halt = True

    def report(self,step,g_norm):
        norm_rep = "Gradient norm {:.4f}".format(g_norm)
        # Profiler
        secs = time.time() - self.prev_time
        num_steps = step - self.prev_steps
        speed = num_steps*self.batch_size / float(secs)
        self.prev_steps = step
        self.prev_time = time.time()
        speed_rep = "Speed: {:.4f} steps/sec".format(speed)
        # Objective
        train_obj = self.eval_obj(self.train)
        dev_obj = self.eval_obj(self.dev)
        obj_rep = "Train Obj.: {:.4f}, Dev Obj: {:.4f}".format(train_obj[0], dev_obj[0])
        print("{}, {}, {}".format(norm_rep, speed_rep,obj_rep))


    def evaluate(self,data,num_samples,sample=True):
        if sample:
            batch_size = np.minimum(num_samples, self.test_batch_size)
            samples = util.chunk(util.sample(data,num_samples), batch_size)
        else:
            samples = util.chunk(data, self.test_batch_size)

        values = [self.evaluator.evaluate(s) for s in samples]
        return np.nanmean(values)

    def eval_obj(self,data):
        samples = util.sample(data,np.minimum(1000,self.test_batch_size))
        loss = self.fprop(samples, volatile=True).data.cpu().numpy()
        return loss


class Reinforce(SGD):
    def __init__(self, train, dev, model, negative_sampler, evaluator, results_dir, config, state=None):
        assert isinstance(negative_sampler,Policy_Sampler)
        super(Reinforce,self).__init__(train, dev, model, negative_sampler, evaluator, results_dir, config, state=state)
        self.arms = dict()
        self.softmax = nn.Softmax()
        # weight decay factor
        self.delta = 0.1
        self.frozen_model = copy.deepcopy(model)

    def fprop(self, batch, volatile=False):
        s_loss = self.reinforce(batch, False, volatile)
        t_loss = self.reinforce(batch, True, volatile)
        return s_loss + t_loss


    def reinforce(self,batch,is_target,volatile):
        entities = self.ns.batch_targets(batch,self.arms,is_target)
        batch_var = util.get_triples(batch, entities, is_target, volatile=volatile)
        scores = self.frozen_model(*batch_var)
        loss = self.sample(batch,is_target,scores,entities)
        # weight decay
        #self.decay()
        return torch.neg(loss)

    def sample(self,batch,is_target,scores,entities):

        policy = self.softmax(scores[:,1:])
        policy_np = policy.data.cpu().numpy()
        loss = Variable(torch.from_numpy(np.asarray([0],dtype='float32')))
        for count in range(policy_np.shape[0]):
            # sample an action (choose a target)
            positives = self.ns.pos(batch[count], is_target)
            neg_map = {entities[count][i]: i for i in range(len(entities[count]))}
            proj_policy = self.project_policy(batch[count], policy_np[count],neg_map, is_target)
            samples = np.random.choice(entities[count],1,p=proj_policy)
            samples_idx = [neg_map[s] for s in samples]
            #Update arms
            rewards = self.compute_reward(samples,samples_idx,scores[count,0].data.cpu().numpy(),scores[count,1:].data.cpu().numpy())
            if len(rewards.keys()) >0:
                for ind,s in enumerate(rewards.keys()):
                    assert s not in positives
                    loss += policy[count,neg_map[s]]*Variable(torch.from_numpy(np.asarray([rewards[s]],dtype='float32')),requires_grad=False)
        return loss


    def compute_reward(self,actions,actions_idx,pos_score,scores):
        scores_map = {a:-1.*scores[actions_idx[i]] for i,a in enumerate(actions)}
        scores_map['pos'] = -pos_score
        sorted_scores = sorted(scores_map.items(), key=operator.itemgetter(1))
        rewards = dict()
        count = 0
        for a,s in sorted_scores:
            if a=='pos':
                rewards = {a: rewards[a] - count for a in rewards.keys()}
                for a in rewards:
                    self.arms[a] = self.arms.get(a,0.0) - rewards[a]
                return rewards
            rewards[a] = count
            count += 1

    def project_policy(self,ex,policy,ent_map,is_target):
        positives = self.ns.pos(ex, is_target)
        for p in positives:
            if p in ent_map:
                policy[ent_map[p]] = 0.0
        return policy / np.sum(policy)

    def pad(self,samples,val):
        if len(samples)<self.ns.num_samples:
            padding = [val] * (self.ns.num_samples - len(samples))
            samples.extend(padding)
        return samples

    def decay(self):
        for a in self.arms:
            self.arms[a] *= self.delta
