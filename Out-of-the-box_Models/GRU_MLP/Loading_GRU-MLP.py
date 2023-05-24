

# Imports

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import random_split
import numpy as np
import pickle
import random
import time
import os
from prettytable import PrettyTable
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

### Settings

data_path = '../../Data/data2_10000.pkl'
config = {
    'RNN': 'GRU',
    'op_dim': 120,
    'joe_rnn_dim':50,
    'joe_rnn_num_layers': 1,
    'schedule_rnn_dim': 50,
    'schedule_rnn_num_layers': 1,
    'setup_embed_dim': 50,
    'final_linear_1_dim': 200,
    'final_linear_2_dim': 100,
    'final_linear_3_dim': 50,
    'final_linear_4_dim': 20,
    'dropout': 0.05,
}

setup_pad_dim = 60
seed = 0
batch_size_ = 128
device = 'cpu'

os.environ['PYTHONHASHSEED']=str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# Loading Data
file = open(data_path, 'rb')

# dump information to that file
loaded_data = pickle.load(file)

# close the file
file.close()

def reshape_list(ind):
    return loaded_data[6*ind:6*ind+6]

len_data = int(len(loaded_data)/6)

data = list(map(reshape_list, range(len_data)))


# Dataset
class JEDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
        self.jobs = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7', 'j8', 'j9', 'j10', 'j11', 'j12']
        self.machines = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8']
        self.machines_ind = {'m1': 0, 'm2':1, 'm3':2, 'm4':3, 'm5':4, 'm6':5, 'm7':6, 'm8':7}

    def one_hot(self,x):
        if len(x) > 0: 
            x = torch.tensor(x)-1
            x = F.one_hot(x, num_classes=120)
            return x.float()
        else:
            return torch.ones((1,120)).float()

    def ready_time(self, index):
        ready_time = torch.empty(0)
        rt = self.data[index][2]
        for i in self.machines:
            oh_ind_m = F.one_hot(torch.tensor([self.machines_ind[i]]), num_classes=8).float()
            rt_m = torch.Tensor([rt[i]]).view(1,1)
            rt_m = torch.cat((rt_m,oh_ind_m),dim=1)
            ready_time = torch.cat((ready_time,rt_m),dim=0)
        return ready_time
    
    def setup(self, index):
        setup = torch.empty(0)
        s = self.data[index][3]
        for i in s:
            first = self.one_hot([i[0]])
            second = self.one_hot([i[1]])
            s_i = torch.cat((first,second),dim=1)
            setup = torch.cat((setup,s_i),dim=0)
        return setup
    
    def joe(self, index):
        j = self.data[index][1]
        l = [j[i] for i in self.jobs]
        return list(map(self.one_hot,l))
    
    def schedule(self, index):
        m = self.data[index][4]
        l = [m[i] for i in self.machines]
        return list(map(self.one_hot,l))
        
    def makespan(self, index):
        m = self.data[index][5]
        return torch.Tensor(m).mean().view(-1)

    def __getitem__(self, index):
        return self.joe(index), self.ready_time(index), self.setup(index), self.schedule(index), self.makespan(index)

    def __len__(self):
        return len(self.data)
    

# Collate
def collate(batch):

    num_machines = 8
    num_jobs = 12
    len_batch = len(batch) 

    joe_collated = []
    joe_batch = [e[0] for e in batch]
    for i in range(num_jobs):
        job_batch = [jobs[i] for jobs in joe_batch]
        padded = pad_sequence(job_batch, batch_first=True).view(len_batch,-1,120)
        lens = list(map(len, job_batch))
        packed = pack_padded_sequence(padded, lens, batch_first=True, enforce_sorted=False)
        joe_collated.append(packed)

    ready_time_batch = [e[1] for e in batch]
    ready_time_collated = torch.stack(ready_time_batch, 0)

    setup_batch = [e[2] for e in batch]
    auxiliary_pad = torch.empty((setup_pad_dim,2*120))
    setup_batch.append(auxiliary_pad)
    setup_padded = pad_sequence(setup_batch, batch_first=True)
    setup_collated = setup_padded[:-1]

    schedules_collated = []
    schedules_batch = [e[3] for e in batch]
    for i in range(num_machines):
        machine_batch = [machines[i] for machines in schedules_batch]
        padded = pad_sequence(machine_batch, batch_first=True).view(len_batch,-1,120)
        lens = list(map(len, machine_batch))
        packed = pack_padded_sequence(padded, lens, batch_first=True, enforce_sorted=False)
        schedules_collated.append(packed)

    makespan = [e[4] for e in batch]
    makespan_collated = torch.stack(makespan, 0)
    
    return joe_collated, ready_time_collated, setup_collated, schedules_collated, makespan_collated

# Dataset
dataset = JEDataset(data)

# Split
generator1 = torch.Generator().manual_seed(seed)
len_dataset = len(dataset)
len_splits = [int(0.8*len_dataset), int(0.1*len_dataset), len_dataset - (int(0.8*len_dataset) + int(0.1*len_dataset))] # 80-10-10 split
train_set, val_set, test_set = random_split(dataset, len_splits, generator=generator1)

train_set_len, val_set_len, test_set_len = len(train_set), len(val_set), len(test_set)

# DataLoaders for the three sets
def loader(batch_size=64):
    train_dataloader = DataLoader(train_set,
                            batch_size=batch_size,
                            collate_fn=collate,
                            drop_last=False,
                            shuffle=True)
    val_dataloader =  DataLoader(val_set,
                            batch_size=batch_size,
                            collate_fn=collate,
                            drop_last=False,
                            shuffle=False)
    test_dataloader = DataLoader(test_set,
                            batch_size=batch_size,
                            collate_fn=collate,
                            drop_last=False,
                            shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader 



class Model(nn.Module):
    
    def __init__(self, config, device='cpu'):

        super().__init__()

        op_dim = config['op_dim']

        RNN = config['RNN']
        self.RNN = RNN

        joe_rnn_dim = config['joe_rnn_dim']
        joe_rnn_num_layers = config['joe_rnn_num_layers']
        if RNN == 'LSTM':
            self.joe_rnn = nn.LSTM(op_dim, joe_rnn_dim,
                                   joe_rnn_num_layers, batch_first=True)
        elif RNN == 'GRU':
            self.joe_rnn = nn.GRU(op_dim, joe_rnn_dim,
                                  joe_rnn_num_layers, batch_first=True)
        else:
            raise Exception("RNN should be specified!")

        schedule_rnn_dim = config['schedule_rnn_dim']
        schedule_rnn_num_layers = config['schedule_rnn_num_layers']
        self.schedule_rnn_num_layers = schedule_rnn_num_layers

        self.final_schedule_dim = schedule_rnn_dim * schedule_rnn_num_layers
        
        if RNN == 'LSTM':
            self.schedule_rnn = nn.LSTM(
                op_dim, schedule_rnn_dim, schedule_rnn_num_layers, batch_first=True)
            self.c0_linear = nn.Linear(9, self.final_schedule_dim)
            self.h0_linear = nn.Linear(9, self.final_schedule_dim)
        elif RNN == 'GRU':
            self.schedule_rnn = nn.GRU(
                op_dim, schedule_rnn_dim, schedule_rnn_num_layers, batch_first=True)
            self.h0_linear = nn.Linear(9, self.final_schedule_dim)
        else:
            raise Exception("RNN should be specified!")

        setup_embed_dim = config['setup_embed_dim']
        self.setup_linear = nn.Linear(2*op_dim, setup_embed_dim)

        self.act = nn.ReLU()

        self.final_linear_1 = nn.Linear(12*joe_rnn_dim*joe_rnn_num_layers + 8*(1+
            self.final_schedule_dim) + setup_embed_dim, config['final_linear_1_dim'])
        self.final_linear_2 = nn.Linear(
            config['final_linear_1_dim'], config['final_linear_2_dim'])
        self.final_linear_3 = nn.Linear(
            config['final_linear_2_dim'], config['final_linear_3_dim'])
        self.final_linear_4 = nn.Linear(
            config['final_linear_3_dim'], config['final_linear_4_dim'])
        self.final_linear_5 = nn.Linear(config['final_linear_4_dim'], 1)

        self.dropout = round(config.get("dropout", 0.0), 2)

    def forward(self, input):

        bs = len(input[-1])

        joe = input[0]
        jobs = torch.empty(0)
        for job in joe:
            if self.RNN == 'LSTM':
                output, (hn, cn) = self.joe_rnn(job)
            else:
                output, hn = self.joe_rnn(job)
            jobs = torch.cat((jobs, hn.view(bs, -1)), dim=1)

        schedule = torch.empty(0)
        for i in range(len(input[3])):
            if self.RNN == 'LSTM':
                h0 = self.h0_linear(input[1][:, i]).view(
                    self.schedule_rnn_num_layers, bs, -1)
                c0 = self.c0_linear(input[1][:, i]).view(
                    self.schedule_rnn_num_layers, bs, -1)
                output, (hn, cn) = self.schedule_rnn(input[3][i], (h0, c0))
            else: 
                h0 = self.h0_linear(input[1][:, i]).view(
                    self.schedule_rnn_num_layers, bs, -1)                
                output, hn = self.schedule_rnn(input[3][i], h0)
            schedule = torch.cat((schedule, hn.view(bs, -1)), dim=1)        
        schedule = schedule.view(-1, 8, self.final_schedule_dim)

        ready_times = input[1][:, :, 0].view(-1, 8, 1)
        schedule_rt = torch.cat((schedule, ready_times), dim=-1)
        schedule_rt = schedule_rt.view(bs, -1)

        embedded_setup = self.setup_linear(input[2])
        mask = input[2].sum(2)
        mask = torch.sign(mask)
        setup = embedded_setup.mul(mask.view(bs, -1, 1)).sum(1)

        final_embedding = torch.cat((schedule_rt, jobs, setup), dim=1)

        final_embedding_1 = self.final_linear_1(final_embedding)
        final_embedding_1 = self.act(final_embedding_1)
        final_embedding_1 = F.dropout(final_embedding_1, self.dropout, training=self.training) 

        final_embedding_2 = self.final_linear_2(final_embedding_1)
        final_embedding_2 = self.act(final_embedding_2)

        final_embedding_3 = self.final_linear_3(final_embedding_2)
        final_embedding_3 = self.act(final_embedding_3)

        final_embedding_4 = self.final_linear_4(final_embedding_3)
        final_embedding_4 = self.act(final_embedding_4)

        makespan = self.final_linear_5(final_embedding_4)

        return makespan


# Loading the test set
train_dataloader, val_dataloader, test_dataloader = loader(100)
test_set_len = len(test_set)
print('Length of the test set is',  test_set_len)
stored_test = []
for element in enumerate(test_dataloader):
    stored_test.append(element)

# Loading the model
model = Model(config)
model.load_state_dict(torch.load('GRU-MLP.pt'))

## Performance in the test set
model.eval()

test_loss1 = 0
test_loss2 = 0
test_ape_loss = 0

metric1 = nn.MSELoss(reduction='sum')
metric2 = nn.L1Loss(reduction='sum')
metric3 = mean_absolute_percentage_error

for i, input in stored_test:

    prediction = model(input)
    label = input[-1]
    loss1_test = metric1(prediction, label)
    loss2_test = metric2(prediction, label)
    len_minibatch = label.shape[0]
    # mean_absolute_percentage_error(y_true, y_pred)
    ape = len_minibatch * metric3(label.detach().numpy(),prediction.detach().numpy())
    test_loss1 += loss1_test.item()
    test_loss2 += loss2_test.item()
    test_ape_loss += ape

test_loss1 /= test_set_len
test_loss2 /= test_set_len
test_ape_loss /= test_set_len

print('\n Test MSE Score is', test_loss1,
    '\n Test RMSE Score is', torch.sqrt(torch.Tensor([test_loss1])).item(),
    '\n Test MAE Score is', test_loss2,
    '\n Test MAPE Score is', test_ape_loss)