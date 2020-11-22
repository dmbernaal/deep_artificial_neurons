import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

__all__ = ['Neuron', 'DANLayer', 'DAN']

class Neuron(nn.Module):
    """
    Core of a single neural component. This can be thought of as a simple 2-layer ANN
    
    ::param ni: input layer dimension
    ::param nf: intermediate layer dimension
    ::nout_ni: next Neuron input layer dimension
    """
    def __init__(self, ni, nf, nout_ni, bias=True, act_fn=None):
        super(Neuron, self).__init__()
        self._ni = ni
        act_fn = nn.Tanh() if act_fn is None else act_fn
        assert isinstance(act_fn, nn.Module), 'activation must be of nn.Module instance'
        
        lin1 = nn.Linear(ni, nf, bias=bias)
        lin2 = nn.Linear(nf, nf, bias=bias)
        lin3 = nn.Linear(nf, nout_ni, bias=bias)
        
        bn1 = nn.BatchNorm1d(ni)
        bn2 = nn.BatchNorm1d(nf)
        
        layers = [bn1, lin1, act_fn, bn2, lin2, act_fn, lin3]
        
        self.core = nn.Sequential(*layers)
        
    def forward(self, x):
        assert x.shape[1] == self._ni, f'input to neuron ({x.shape[1]}) does not match input dimension size: ({self._ni})'
        return self.core(x)
    
class DANLayer(nn.Module):
    """
    A single DANLayer is composed of n number of Neurons.
    
    ::param neurons: number of neurons in given layer
    ::param n_ni: dendritic dimension, intermediate layers take concatenated output from previous layer
    ::param n_nf: intermediate neuron dimension
    ::param n_out_ni: axonal dimension
    ::param p_neurons: number of neurons in previous layer 
    """
    def __init__(self, neurons, n_ni, n_nf, n_nout_ni, p_neurons=1, bias=True, act_fn=None):
        super(DANLayer, self).__init__()
        self._n = neurons
        self._ni = n_ni*p_neurons
        self.neurons = nn.ModuleList([Neuron(n_ni*p_neurons, n_nf, n_nout_ni, bias=bias, act_fn=act_fn) for _ in range(neurons)])
        
    def forward(self, x):
        """
        ::param x: dimensions -> (batch, features)
        for intermediate layers, features will be the concatenated output
        """
        b, f = x.size()
        assert f==self._ni, f'input dimension ({f}) do not match neuron input dimension ({self._ni})'
        
        outs = [self.neurons[i](x) for i in range(self._n)]
        return torch.cat(outs, dim=1)
    
class DAN(nn.Module):
    """
    Hardcodes DAN architecture. This follows the paper https://arxiv.org/pdf/2011.07035.pdf
    A simple 3 layer DAN
    ::param ni: number of dimension for first DANLayer
    ::param num_classes: number of classes to predict
    """
    def __init__(self, ni, num_classes, act=None):
        super(DAN, self).__init__()
        """
        NOTE: n_out_ni must match next layers n_ni
        TODO: make this dynamic
        """
        self.dan1 = DANLayer(neurons=3, n_ni=ni, n_nf=50, n_nout_ni=100, p_neurons=1, act_fn=act)
        self.dan2 = DANLayer(neurons=2, n_ni=100, n_nf=50, n_nout_ni=25, p_neurons=3, act_fn=act)
        self.dan3 = DANLayer(neurons=1, n_ni=25, n_nf=10, n_nout_ni=num_classes, p_neurons=2, act_fn=act)
        
    def forward(self, x):
        """
        ::param x: shape (batch, features)
        """
        x = self.dan1(x)
        x = self.dan2(x)
        return self.dan3(x)