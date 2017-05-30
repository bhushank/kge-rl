__author__=  'Bhushan Kotnis'

#Index file names
entity_ind = 'entities.cpkl'
rel_ind = 'relations.cpkl'

'''SGD Batch Size'''
batch_size = 4000
test_batch_size = 4000

'''Negatives'''
num_train_negs = 10
num_dev_negs = 1000
num_test_negs = float('inf')


'''Report and Save model'''
report_steps = 10


'''Early Stopping'''
early_stop_counter = 5
patience = 10
num_epochs=50

'''Dataset details'''
fb15k_rels = 1345
fb15k_ents = 14951

wn_rels = 18
wn_ents = 40943

cat_file='/home/mitarb/kotnis/Code/kge-rl/entity_cat.cpkl'
