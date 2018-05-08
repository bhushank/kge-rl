import torch
import torch.nn as nn
import util
import numpy as np

class KGE(nn.Module):
    def __init__(self,n_ents,n_rels,ent_dim,rel_dim,max_norm=False):
        super(KGE,self).__init__()
        self.num_ents = n_ents
        if max_norm:
            self.entities = nn.Embedding(n_ents,ent_dim,max_norm=1)
        else:
            self.entities = nn.Embedding(n_ents, ent_dim)

        self.rels = nn.Embedding(n_rels,rel_dim)
        self.init()

    def forward(self,sources,targets,rels):
        raise NotImplementedError('Abstract method')

    def predict(self, batch,negs=None,is_target=True,is_pad=False):
        sources,targets, rels = util.get_triples(batch,negs,
                                                 is_target=is_target,volatile=True,is_pad=is_pad)
        return self.forward(sources,targets,rels)

    def init(self):
        self.entities.weight.data.uniform_(-0.1, 0.1)
        self.rels.weight.data.uniform_(-0.1, 0.1)

    def broadcast(self,sources,targets,rels):
        # PyTorch 0.1.11 does not support broadcasting
        rels = rels.unsqueeze(1)
        if sources.size()[1] > targets.size()[1]:
            rels = rels.expand_as(sources)
            targets = targets.expand_as(sources)
        else:
            rels = rels.expand_as(targets)
            sources = sources.expand_as(targets)
        return sources,targets,rels

    def inner_prod(self,s,r,t):
        r = r.unsqueeze(1).expand_as(t)
        prod = torch.mul(r,t).transpose(1,2)
        score = torch.bmm(s, prod)
        return score

    def all_entity_vectors(self):
        var = util.to_var(range(self.num_ents),volatile=True)
        entities = self.entities(var).data.cpu().numpy()
        return entities

    def entity_vectors(self,ids):
        var = util.to_var(ids,volatile=True)
        vector = self.entities(var).data.cpu().numpy()
        return vector


    def relation_vectors(self,ids):
        var = util.to_var(ids,volatile=True)
        vector = self.rels(var).data.cpu().numpy()
        return vector

class Rescal(KGE):

    def __init__(self,n_ents,n_rels,ent_dim):
        super(Rescal,self).__init__(n_ents,n_rels,ent_dim,ent_dim*ent_dim,max_norm=True)
        self.dim = ent_dim
        print("Initializing RESCAL model")

    def init(self):
        #self.entities.weight.data.normal_(std=0.1)
        #self.rels.weight.data.normal_(std=0.1)
        self.entities.weight.data.uniform_(-1., 1.)
        self.rels.weight.data.uniform_(-1., 1.)

    def forward(self,sources,targets,rels):
        sources = self.entities(sources)
        targets = self.entities(targets)
        rels = self.rels(rels)
        #Reshape rels
        rels = rels.view(-1,self.dim,self.dim)
        #score = x_s^T W_r x_t
        out = torch.bmm(torch.bmm(sources,rels),torch.transpose(targets,1,2))
        # First element is positive, rest are negatives
        out = out.view(-1,out.size()[1]*out.size()[2])
        return out

    def output(self,entities,rels,is_target):
        '''
        Given source and rels output the target or given target and rels output the source vector
        :param entities: source or target entity ids
        :param rels: rel ids
        :param is_target: True for predicting targets
        :return: 
        '''
        entities = self.entities(util.to_var(entities,True)).unsqueeze(2)
        rels = self.rels(util.to_var(rels,True))
        # Reshape rels
        rels = rels.view(-1, self.dim, self.dim)
        if is_target:
            out = torch.bmm(torch.transpose(entities, 1, 2),rels)
        else:
            out = torch.bmm(rels, entities)
        out = out.view(-1, out.size()[1] * out.size()[2])
        return out.data.cpu().numpy()


class TransE(KGE):
    def __init__(self, n_ents, n_rels, ent_dim):
        super(TransE, self).__init__(n_ents, n_rels, ent_dim, ent_dim,max_norm=True)
        print("Initializing TransE model")

    def forward(self,sources,targets,rels):
        sources = self.entities(sources)
        targets = self.entities(targets)
        rels = self.rels(rels)
        sources,targets,rels = self.broadcast(sources,targets,rels)
        # score = -||x_s + x_r - x_t||_2
        d = sources + rels - targets
        #d = torch.abs(d)
        d = torch.mul(d, d)
        d = torch.sum(d,2)
        return torch.neg(d)

    def output(self,entities,rels,is_target):
        entities = self.entities(util.to_var(entities, True)).unsqueeze(2)
        rels = self.rels(util.to_var(rels, True))
        if is_target:
            out = entities + rels
        else:
            out = entities - rels
        return out.view(-1,out.size()[1] * out.size()[2]).data.cpu().numpy()


class Distmult(KGE):
    def __init__(self, n_ents, n_rels, ent_dim):
        super(Distmult, self).__init__(n_ents, n_rels, ent_dim, ent_dim,max_norm=True)
        print("Initializing Distmult model")

    def forward(self, sources, targets, rels):
        sources = self.entities(sources)
        targets = self.entities(targets)
        rels = self.rels(rels)
        # score = x_s^T Diag(W_r) x_t
        if sources.size()[1] > targets.size()[1]:
            out = self.inner_prod(sources,rels,targets).squeeze(2)
        else:
            out = self.inner_prod(targets, rels, sources).squeeze(2)

        return out

    def output(self,entities,rels,is_target):
        entities = self.entities(util.to_var(entities, True)).unsqueeze(2)
        rels = self.rels(util.to_var(rels, True))
        out = torch.mul(entities,rels)
        return out.squeeze(2).data.cpu().numpy()

    def init(self):
        #self.entities.weight.data.normal_(std=0.1)
        #self.rels.weight.data.normal_(std=0.1)
        self.entities.weight.data.normal_()
        self.rels.weight.data.normal_()

class ComplEx(KGE):
    def __init__(self, n_ents, n_rels, ent_dim):
        super(ComplEx, self).__init__(n_ents, n_rels, ent_dim, ent_dim,max_norm=True)
        self.entities_i = nn.Embedding(n_ents,ent_dim,max_norm=1)
        self.rels_i = nn.Embedding(n_rels,ent_dim)
        self.init_all()
        print("Initializing ComplEx model")

    def init_all(self):
        self.entities.weight.data.normal_()
        self.rels.weight.data.normal_()
        self.entities_i.weight.data.normal_()
        self.rels_i.weight.data.normal_()


    def forward(self,sources,targets,rels):


        # score = <w_r,e_s,e_o> + <w_r,e_s_i,e_o_i> + <w_r_i,e_s,e_o_i> - <w_r_i,e_s_i,e_o>
        if sources.size()[1] > targets.size()[1]:
            return self.complex(targets, rels, sources)

        return self.complex(sources, rels, targets)

    def complex(self, sources, rels, targets):
        sources_i = self.entities_i(sources)
        targets_i = self.entities_i(targets)
        rels_i = self.rels_i(rels)
        sources = self.entities(sources)
        targets = self.entities(targets)
        rels = self.rels(rels)

        out = self.inner_prod(targets, rels, sources) \
              + self.inner_prod(targets_i, rels, sources_i) \
              + self.inner_prod(targets_i, rels_i, sources) \
              - self.inner_prod(targets, rels_i, sources_i)
        out = out.squeeze(2)
        return out

    def output(self,entities,rels,is_target):
        entities_i = self.entities_i(util.to_var(entities, True)).unsqueeze(2)
        rels_i = self.rels_i(util.to_var(rels, True))
        entities = self.entities(util.to_var(entities, True)).unsqueeze(2)
        rels = self.rels(util.to_var(rels, True))
        if is_target:
            out = torch.mul(entities,rels) + torch.mul(entities_i,rels) \
                  + torch.mul(rels_i,entities) - torch.mul(rels_i,entities_i)
        else:
            out = torch.mul(entities, rels) + torch.mul(entities_i, rels) \
                  + torch.mul(rels_i, entities_i) - torch.mul(rels_i, entities)
        return out.squeeze(2).data.cpu().numpy()


    def entity_vectors(self, ids):
        var = util.to_var(ids, volatile=True)
        ents_r = self.entities(var).data.cpu().numpy()
        ents_i = self.entities_i(var).data.cpu().numpy()
        ents_v = np.concatenate((ents_r, ents_i), axis=1)
        return ents_v


    def relation_vectors(self,ids):
        var = util.to_var(ids,volatile=True)
        rels_r = self.rels(var).data.cpu().numpy()
        rels_i = self.rels_i(var).data.cpu().numpy()
        rels_v = np.concatenate((rels_r,rels_i),axis=1)
        return rels_v