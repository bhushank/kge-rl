import numpy as np
import util
import time
import constants
import torch
from torch.optim import Adam
import os
from torch import nn

class SGD(object):
    def __init__(self,train,dev,model,negative_sampler,evaluator,results_dir,config):
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
        self.batch_size = config.get('batch_size',constants.batch_size)
        print("lr: {:.4f}, l2: {:.5f}, batch_size: {}".format(lr,l2,self.batch_size))
        self.optim = Adam(model.parameters(),lr=lr,weight_decay=l2)

        #Report and Early Stopping Params
        self.prev_score = evaluator.init_score
        self.early_stop_counter = constants.early_stop_counter
        self.patience = constants.patience
        self.num_epochs = config.get('num_epochs',constants.num_epochs)

        self.report_steps = constants.report_steps
        self.test_batch_size = config.get('test_batch_size',constants.test_batch_size)
        self.halt = False

        self.prev_steps = 0
        self.prev_time = time.time()

        #Loss
        self.mm = nn.MarginRankingLoss(margin=1)
        self.logistic = nn.SoftMarginLoss()


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
                g_norm = torch.nn.utils.clip_grad_norm(self.model.parameters(), 3.0)
                self.optim.step()
                if step % self.report_steps == 0 and step!=0:
                    self.report(step,g_norm)
            self.prev_steps=0
            self.prev_time=time.time()
            end = time.time()
            mins = int(end - start)/60
            secs = int(end - start)%60
            print("Epoch {}. Took {} minutes {} seconds".format(epoch+1,mins,secs))
            self.save()
            if self.halt:
                return

    def fprop(self,batch,volatile=False):
        source_negs = self.ns.batch_sample(batch, False)
        target_negs = self.ns.batch_sample(batch, True)
        s_batch = util.get_triples(batch,source_negs,False,volatile=volatile)
        t_batch = util.get_triples(batch, target_negs, True,volatile=volatile)
        s_score = self.model(*s_batch)
        t_score = self.model(*t_batch)
        # score at index 0 is positive
        if self.model_name in {'transE', 'rescal'}:
            return self.max_margin(s_score,volatile)+ self.max_margin(t_score,volatile)
        return self.nll(s_score,volatile) + self.nll(t_score,volatile)


    def max_margin(self,scores,volatile):
        y = util.to_var(np.ones(scores.size()[0],dtype='float32'),volatile=volatile)
        loss = self.mm(scores[:,0],scores[:,1],y)
        for i in range(2,scores.size()[1]):
            loss += self.mm(scores[:,0],scores[:,i],y)
        return loss/(scores.size()[1]-1.)

    def nll(self,scores,volatile):
        y_pos = util.to_var(np.ones(scores.size()[0],dtype='float32'),volatile=volatile)
        y_neg = util.to_var(-1.*np.ones(scores.size()[0], dtype='float32'),volatile=volatile)
        loss = self.logistic(scores[:, 0],y_pos)
        for i in range(1,scores.size()[1]):
            loss += self.logistic(scores[:, i], y_neg)
        return loss/scores.size()[1]


    def save(self):
        curr_score = self.evaluate(self.dev,self.test_batch_size,True)
        print("Current Score: {}, Previous Score: {}".format(curr_score,self.prev_score))
        if self.evaluator.comparator(curr_score, self.prev_score):
            print("Saving params...\n")
            torch.save(self.model.state_dict(), os.path.join(self.results_dir,'{}_params.pt'.format(self.model_name)))
            self.prev_score = curr_score
            # Reset early stop counter
            self.early_stop_counter = self.patience
        else:
            self.early_stop_counter -= 1
            print("New params worse than current, skip saving...\n")

        if self.early_stop_counter <= 0:
            self.halt = True

    def report(self,step,g_norm):
        norm_rep = "Gradient norm {:.3f}".format(g_norm)
        # Profiler
        secs = time.time() - self.prev_time
        num_steps = step - self.prev_steps
        speed = num_steps*self.batch_size / float(secs)
        self.prev_steps = step
        self.prev_time = time.time()
        speed_rep = "Speed: {:.3f} steps/sec".format(speed)
        # Objective
        train_obj = self.eval_obj(self.train)
        dev_obj = self.eval_obj(self.dev)
        obj_rep = "Train Obj: {:.4f}, Dev Obj: {:.4f}".format(train_obj[0], dev_obj[0])
        print("{},{},{}".format(norm_rep, speed_rep,obj_rep))


    def evaluate(self,data,num_samples,sample=True):
        if sample:
            batch_size = np.minimum(num_samples, self.test_batch_size)
            samples = util.chunk(util.sample(data,num_samples), batch_size)
        else:
            samples = util.chunk(data, self.test_batch_size)

        values = [self.evaluator.evaluate(s,num_negs=0) for s in samples]
        return np.nanmean(values)

    def eval_obj(self,data):
        samples = util.sample(data,np.minimum(1000,self.test_batch_size))
        loss = self.fprop(samples, volatile=True).data.cpu().numpy()
        return loss