import models
from data_loader import Index
import os
import torch
import pickle as pickle
import numpy as np

def get_model(model,n_ents,n_rels,ent_dim):
    if model == 'rescal':
        return models.Rescal(n_ents, n_rels,ent_dim)
    elif model == 'transE':
        return models.TransE(n_ents, n_rels, ent_dim)
    elif model == 'distmult':
        return models.Distmult(n_ents, n_rels, ent_dim)
    elif model == 'complex':
        return models.ComplEx(n_ents, n_rels, ent_dim)
    else:
        raise NotImplementedError("Model {} not implemented".format(model))


def save_embeddings(results_dir,model_name,is_cpu=True,ent_dim=100):
    index = Index()
    index.load_index(results_dir)
    n_ents = index.ent_vocab_size()
    n_rels = index.rel_vocab_size()
    model = get_model(model_name,n_ents,n_rels,ent_dim)
    params_path = os.path.join(results_dir, '{}_params.pt'.format(model_name))
    if is_cpu:
        model_params = torch.load(params_path, map_location=lambda storage, loc: storage)
    else:
        model_params = torch.load(params_path)

    model.load_state_dict(model_params)

    relation_embeddings = dict()
    entity_embeddings = dict()
    
    for rel in index.rel_index:
        relation_embeddings[rel] = model.relation_vectors([index.rel_to_ind(rel)])

    for ent in index.ent_index:
        entity_embeddings[ent] = model.entity_vectors([index.ent_to_ind(ent)])
    
    pickle.dump(relation_embeddings,open(results_dir+'/relation_emb.cpkl','wb'))
    pickle.dump(entity_embeddings, open(results_dir + '/entity_emb.cpkl', 'wb'))


