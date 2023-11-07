import torch
import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self,input_dim = None, output_dim = None, **kwargs):
        super(MLPModel,self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = kwargs.get('num_layers', 2)
        self.hidden_size = kwargs.get('hidden_size', 2048)

        self.begin = nn.Linear(self.input_dim, self.hidden_size)

        self.hidden = nn.Sequential()

        for i in range(self.num_layers):
            self.hidden.append(nn.Linear(self.hidden_size,self.hidden_size))
            self.hidden.append(nn.ReLU())

        self.end = nn.Linear(self.hidden_size,self.output_dim)

    def batch_collate_fn(self, samples):
        if isinstance(samples[0][0],dict):
            x = [s[0] for s in samples]
        else:
            x = torch.stack([s[0] for s in samples]).float()
        label = torch.stack([s[1] for s in samples])
        return x, label
    
    def forward(self, net_input):
        x1 = self.begin(net_input)
        x1 = torch.relu(x1)

        x2 = self.hidden(x1)
        x2 = torch.relu(x2)

        x3 = self.end(x2)

        return x3
