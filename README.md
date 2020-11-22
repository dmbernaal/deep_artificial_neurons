# Deep Artificial Neurons
My take on ```Deep Artificial Neurons``` from paper: https://arxiv.org/pdf/2011.07035.pdf

# Abstract from paper
*Neurons in real brains are enormously complex computational units. Among
other things, they’re responsible for transforming inbound electro-chemical vectors
into outbound action potentials, updating the strengths of intermediate synapses,
regulating their own internal states, and modulating the behavior of other nearby
neurons. One could argue that these cells are the only things exhibiting any
semblance of real intelligence. It is odd, therefore, that the machine learning
community has, for so long, relied upon the assumption that this complexity can be
reduced to a simple sum and fire operation. We ask, might there be some benefit to
substantially increasing the computational power of individual neurons in artificial
systems? To answer this question, we introduce Deep Artificial Neurons (DANs),
which are themselves realized as deep neural networks. Conceptually, we embed
DANs inside each node of a traditional neural network, and we connect these
neurons at multiple synaptic sites, thereby vectorizing the connections between
pairs of cells. We demonstrate that it is possible to meta-learn a single parameter
vector, which we dub a neuronal phenotype, shared by all DANs in the network,
which facilitates a meta-objective during deployment. Here, we isolate continual
learning as our meta-objective, and we show that a suitable neuronal phenotype can
endow a single network with an innate ability to update its synapses with minimal
forgetting, using standard backpropagation, without experience replay, nor separate
wake/sleep phases. We demonstrate this ability on sequential non-linear regression
tasks.*

## !NOTE!
I am not an author of the paper. Proper citations are provided below (or link above). This is just my understanding of the Architecture behind the proposed model.

## Neuron
```python
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
```

## DAN Layer
```python
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
```

## DAN Model
```python
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
```

## Citation
```
@article{
  title={Continual Learning with Deep Artificial Neurons},
  author={Blake Camp},
  journal={arXiv arXiv:2011.07035},
  year={2020}
}

@article{
  title={Continual Learning with Deep Artificial Neurons},
  author={Jaya Krishna Mandivarapu},
  journal={arXiv arXiv:2011.07035},
  year={2020}
}

@article{
  title={Continual Learning with Deep Artificial Neurons},
  author={Rolando Estrada},
  journal={arXiv arXiv:2011.07035},
  year={2020}
}

```