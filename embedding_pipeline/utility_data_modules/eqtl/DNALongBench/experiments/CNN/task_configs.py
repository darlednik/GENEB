import torch
import torch.nn as nn
import torch.optim as optim
from cnn import SimpleCNN, CNN

def get_model(args):
    task_name = args.task_name
    if task_name == 'contact_map_prediction':
        model = CNN()
    elif task_name == 'regulatory_sequence_activity':
        if args.subset == 'human':
            model =  SimpleCNN(channels = [16,64,256,1024], output_shape=(896,5313), input_length=196608, use_max_pool=True, kernel_sizes = [25,15,15,15],  strides=[2,2,2,2],task='RSAP')
        else:
            model =  SimpleCNN(channels = [16,64,256,1024], output_shape=(896,1643), input_length=196608, use_max_pool=True, kernel_sizes = [25,15,15,15],  strides=[2,2,2,2],task='RSAP')
    elif task_name == 'transcription_initiation_signal_prediction':
        model = SimpleCNN(channels = [16,64,256,1024], output_shape=(100000,10), input_length=100000)
    elif task_name == 'enhancer_target_gene_prediction':
        model = SimpleCNN(channels = [128,64,32],input_length=450000, task='ETGP')
    elif task_name == 'eqtl_prediction':
        model = SimpleCNN(channels = [128,64,32],input_length=450000, task='eQTLP')
    return model

def get_configs(task_name):
    if task_name == 'contact_map_prediction':
        loss = nn.MSELoss()
        metric = 'pcc'
    elif task_name == 'regulatory_sequence_activity':
        loss = nn.MSELoss()
        metric = 'pcc'
    elif task_name == 'transcription_initiation_signal_prediction':
        loss = nn.MSELoss()
        metric = 'pcc'
    elif task_name == 'enhancer_target_gene_prediction':
        loss = nn.CrossEntropyLoss()
        metric = 'auroc'
    elif task_name == 'eqtl_prediction':
        loss = nn.CrossEntropyLoss()
        metric = 'auroc'
    return loss, metric