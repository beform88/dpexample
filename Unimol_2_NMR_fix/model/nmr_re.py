import torch
import torch.nn as nn


class NMRREModel(nn.Module):
    def __init__(self,input_dim = None, output_dim = None, **kwargs):
        super(NMRREModel,self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = 1
        self.hidden_size = 8

        self.begin = nn.Linear(self.input_dim, 120)
        self.drop = nn.Dropout(0.15)
        self.hidden = nn.Sequential()

        self.hidden.append(nn.Linear(120,self.hidden_size,))
        self.hidden.append(nn.Softmax(dim = 0))

        self.end = nn.Linear(self.hidden_size,self.output_dim)

    def batch_collate_fn(self, samples):
        x = torch.stack([s[0] for s in samples]).float()
        label = torch.stack([s[1] for s in samples])
        return x, label
    
    def forward(self, net_input):
        x1 = self.begin(net_input)
        x1 = self.drop(x1)
        x1 = torch.softmax(x1,dim = 0)

        x2 = self.hidden(x1)

        x3 = self.end(x2)

        return x3
