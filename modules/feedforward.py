import torch
import torch.nn as nn

class FeedForwardNet(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dims, 
                 output_dim, 
                 activation='relu', 
                 use_batchnorm=True):
        """ A feedforward network.

        Parameters:
        input_dim (int): The size of the input 
        hidden_dims (list): The sizes of all the hidden layers
        output_dim (int): The size of the output
        activation (string): The activation function to be used (default: "relu")
        use_batchnorm (bool): Whether to apply batch normalization (default: True) 
        """
        super(FeedForwardNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.use_batchnorm = use_batchnorm
        
        layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dims[i-1], hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
        
        if len (hidden_dims) > 0:
            layers.append (nn.Linear(hidden_dims[-1], output_dim))
        else:
            layers.append (nn.Linear(input_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        """ The forward method transforms the input and returns the output.
        """
        x = self.net(x)
        return x