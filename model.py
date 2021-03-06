#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
class Spatio_Block(nn.Module):
    def __init__(self, n_inputs, n_channels):
        super(Spatio_Block, self).__init__()
        self.relu = nn.ReLU()
        
        ##########################################
        self.net_a = weight_norm(nn.Conv1d(n_inputs, 128,1))
        self.net_b = weight_norm(nn.Conv1d(128, 128, 1))#
        self.net_c = weight_norm(nn.Linear(n_channels,128))
        self.net_d = weight_norm(nn.Linear(128,128))
        self.net_e = weight_norm(nn.Conv1d(128, n_inputs, 1))
        self.net_f = weight_norm(nn.Linear(128,n_channels)) 
           
        ###########################################
        self.net = nn.Sequential(self.net_a, 
                                 # self.net_b,
                                 self.net_c,
                                 # self.net_d,
                                 self.net_e,
                                 self.net_f,
                                 self.relu,)
        self.init_weights()
    def init_weights(self):
        self.net_a.weight.data.normal_(0, 0.01)
        self.net_b.weight.data.normal_(0, 0.01)
        self.net_c.weight.data.normal_(0, 0.01)
        self.net_d.weight.data.normal_(0, 0.01)
        self.net_e.weight.data.normal_(0, 0.01)
        self.net_f.weight.data.normal_(0, 0.01)     
    def forward(self, x):
        y = self.net(x)
        return y 
class Spatio_ConvNet(nn.Module):
    def __init__(self, n_inputs, n_channels):
        super(Spatio_ConvNet, self).__init__()
        self.relu = nn.ReLU()
        
        ##########################################
        self.net_a = weight_norm(nn.Conv1d(n_inputs, 128, 1))
        self.net_b = weight_norm(nn.Conv1d(128, 128, 1))#
        self.net_c = weight_norm(nn.Linear(n_channels,128))
        self.net_d = weight_norm(nn.Linear(128,128))#
        self.pool1 = nn.MaxPool2d(2,2)
        ###########################################
        self.net_e = weight_norm(nn.Conv1d(64, 64, 1))#
        self.net_f = weight_norm(nn.Conv1d(64, 32, 1))
        self.net_g = weight_norm(nn.Linear(64, 64))#
        self.net_h = weight_norm(nn.Linear(64, 32))
        self.pool2 = nn.MaxPool2d(2,2)
        ###########################################
        self.net_i = weight_norm(nn.Conv1d(16, 16, 1))#
        self.net_j = weight_norm(nn.Conv1d(16, 4, 1))
        self.net_k = weight_norm(nn.Linear(16, 16))#
        self.net_l = weight_norm(nn.Linear(16, 4))
        
        
        self.net = nn.Sequential(self.net_a,self.net_b,self.net_c,self.net_d,self.relu,self.pool1,
                                 self.net_e,self.net_f,self.net_g,self.net_h,self.relu,self.pool2,
                                 self.net_i,self.net_j,self.net_k,self.net_l,)
                                                      
        self.fc = weight_norm(nn.Linear(16, 1))
        self.init_weights()
    def init_weights(self):
        self.net_a.weight.data.normal_(0, 0.01)
        self.net_b.weight.data.normal_(0, 0.01)
        self.net_c.weight.data.normal_(0, 0.01)
        self.net_d.weight.data.normal_(0, 0.01)
        
        self.net_e.weight.data.normal_(0, 0.01)
        self.net_f.weight.data.normal_(0, 0.01)
        self.net_g.weight.data.normal_(0, 0.01)
        self.net_h.weight.data.normal_(0, 0.01)
        
        self.net_i.weight.data.normal_(0, 0.01)
        self.net_j.weight.data.normal_(0, 0.01)
        self.net_k.weight.data.normal_(0, 0.01)
        self.net_l.weight.data.normal_(0, 0.01)
        self.fc.weight.data.normal_(0, 0.01)
    def forward(self, x):
        y = self.net(x).squeeze(2)
        y = self.fc(y.view(-1, 16))
        y = self.relu(y)
        return y
class Temporal_and_Spatio_Block(nn.Module):
    def __init__(self, n_inputs, input_length, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(Temporal_and_Spatio_Block, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,stride=stride, padding=padding, dilation=dilation))       
        self.chomp1 = Chomp1d(padding)  
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        #-------------------------------------------------------#
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
                                                                                                  
        self.temporal_net = nn.Sequential(self.conv1, self.chomp1,self.relu1, self.dropout1,)
                                          # self.conv2, self.chomp2,self.relu2, self.dropout2,)
        self.spatio_net = Spatio_Block(n_inputs,input_length)
                                                
        self.relu = nn.ReLU()
        self.init_weights()
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
    def forward(self, x):
        y1 = self.temporal_net(x)  
        y2 = self.spatio_net(x)
        return self.relu(x + y1 + y2)
        
class Temporal_and_Spatio_ConvNet(nn.Module):
    def __init__(self, num_inputs, input_length, num_channels, kernel_size, dropout):
        super(Temporal_and_Spatio_ConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i   
            in_channels = num_inputs if i == 0 else num_channels[i-1]           
            out_channels = num_channels[i]  #
            layers += [Temporal_and_Spatio_Block(in_channels,input_length, out_channels, kernel_size,stride=1, 
                            dilation=dilation_size,padding=(kernel_size-1) * dilation_size, dropout=dropout)]
    
        self.temporal_and_spatio_network = nn.Sequential(*layers)

    def forward(self, x):                                                                                           
        return self.temporal_and_spatio_network(x)
class TnS_net(nn.Module):
    def __init__(self, input_channel, input_length, output_channel,num_channels, kernel_size, dropout):
        super(TnS_net, self).__init__()
        self.TnS = Temporal_and_Spatio_ConvNet(input_channel, input_length,num_channels, kernel_size, dropout)
        self.Spatio_net = Spatio_ConvNet(input_channel,input_length)
        self.linear = nn.Linear(num_channels[-1], output_channel)
    def forward(self, x):
        y = self.TnS(x)  # input should have dimension (N, C, L)
        y1 = self.linear(y[:, :, -1]) 
        y2 = self.Spatio_net(x)
        return torch.sigmoid(y1 + y2)
        