import numpy as np
import util
import time
import constants
import torch
from torch.optim import Adam
import os
from torch import nn
from sys import stdout

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
        print("lr:{:.3f}, l2:{:.3f}, batch_size:{}".format(lr,l2,self.batch_size))
        self.optimizer = Adam(lr=lr,weight_decay=l2)

        #Report and Early Stopping Params
        self.prev_score = evaluator.init_score
        self.early_stop_counter = constants.early_stop_counter
        self.patience = constants.patience
        self.num_epochs = config.get('num_epochs',constants.num_epochs)

        self.report_steps = constants.report_steps
        self.save_epochs = constants.save_epochs
        self.halt = False

        self.prev_steps = 0
        self.prev_time = time.time()

        #Loss
        self.criterion = nn.MultiMarginLoss()


    def minimize(self):
        print("Training...")
        for epoch in range(self.num_epochs):
            train_cp = list(self.train)
            np.random.shuffle(train_cp)
            batches = util.chunk(train_cp, self.batch_size)
            for step,batch in enumerate(batches):
                loss = self.fprop(batch)
                loss.backward()
                g_norm = torch.nn.utils.clip_grad_norm(self.model.parameters(), 3.0)
                self.optimizer.step()
                if step % self.report_steps == 0:
                    self.report(step,g_norm)

            self.prev_steps=0
            self.prev_time=time.time()
            print("Number of Epochs {}".format(epoch))
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
        y = util.to_var(np.zeros(len(batch), dtype='float32'),volatile=volatile)
        loss = self.criterion(s_score,y) + self.criterion(t_score,y)
        return loss

    def save(self):
        curr_score = self.evaluate(self.dev, True,num_samples=5*constants.test_batch_size)
        if self.evaluator.comparator(curr_score, self.prev_score):
            print("Saving params...")
            torch.save(self.model.state_dict(), os.path.join(self.results_dir,'{}_params.pt'.format(self.model_name)))
            self.prev_score = curr_score
            # Reset early stop counter
            self.early_stop_counter = self.patience
        else:
            self.early_stop_counter -= 1
            print("New params worse than current, skip saving...")

        if self.early_stop_counter <= 0:
            self.halt = True

    def report(self,step,g_norm):
        norm_rep = "Gradient norm {:.3f}".format(g_norm)
        # Profiler
        secs = time.time() - self.prev_time
        num_steps = step - self.prev_steps
        speed = num_steps / float(secs)
        self.prev_steps = step
        self.prev_time = time.time()
        speed_rep = "Speed: {:.2f} steps/sec".format(speed)
        # Objective
        train_obj = self.eval_obj(self.train)
        dev_obj = self.eval_obj(self.dev)
        obj_rep = "Train Obj: {:.3f}, Dev Obj: {:.3f}".format(train_obj[0], dev_obj[0])
        # Performance
        train_val = self.evaluate(self.train, sample=True)
        dev_val = self.evaluate(self.dev, sample=True)
        metric = self.evaluator.metric_name
        eval_rep = "Train {} {:.3f}, Dev {} {:.3f}".format(metric, train_val, metric, dev_val)
        stdout.write("{},{},{}, {}".format(norm_rep, speed_rep, obj_rep, eval_rep))
        stdout.flush()

    def evaluate(self,data,sample=True,num_samples=constants.test_batch_size):
        if sample:
            batch_size = np.minimum(num_samples, constants.test_batch_size)
            samples = util.chunk(util.sample(data,num_samples), batch_size)
        else:
            samples = util.chunk(data, constants.test_batch_size)

        values = [self.evaluator.evaluate(s) for s in samples]
        return np.nanmean(values)

    def eval_obj(self,data):
        samples = util.sample(data,constants.test_batch_size)
        loss = self.fprop(samples, volatile=True).data.cpu().numpy()
        return loss