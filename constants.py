__author__=  'Bhushan Kotnis'

#Index file names
entity_ind = 'entities.cpkl'
rel_ind = 'relations.cpkl'


'''SGD Batch Size'''
batch_size = 4000
test_batch_size = 5000

'''Negatives'''
num_train_negs = 10
num_dev_negs = 100
num_test_negs = float('inf')


'''Report and Save model'''
report_steps = 20
save_epochs = 1

'''Early Stopping'''
early_stop_counter = 3
patience = 3
num_epochs=1000


'''Position of positive sample'''
pos_position=-1
