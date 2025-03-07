from flax import linen as nn
from flax.linen import Dense
from typing import Sequence


class MLP(nn.Module):
    """
    --- multi-layer perceptron --- 
    args:
        input_dim: int - number of input features
        output_dim: int - number of output features
        hidden_layers: Sequence[int] - list of hidden layer sizes
    """
    input_dim: int = 5
    output_dim: int = 5
    hidden_layers: Sequence[int] = (256, 256, 256) 

    def setup(self):
        self.layers = [Dense(features=hidden_dim) for hidden_dim in self.hidden_layers]
        self.output_layer = Dense(features=self.output_dim)

    @nn.remat
    def __call__(self, X):
        for layer in self.layers:
            X = layer(X)
            X = nn.relu(X)
        X = self.output_layer(X)
        return X