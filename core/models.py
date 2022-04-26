import jax
import haiku as hk
from jax.flatten_util import ravel_pytree
from .utils import ravel_pytree_ as ravel_fn


def make_mlp_fn(layer_dims, output_dim):
    """Returns a forward function for an MLP of given dimensions."""

    def forward(x):
        """
        Input: [B, C, L]
        Output: [B, output_dim]]
        """

        # flatten input
        x = hk.Flatten()(x) # [B, C, L] -> [B, C*L]

        # hidden layers
        for layer_dim in layer_dims:
            x = hk.Linear(layer_dim)(x) # [B, layer_dim]
            x = jax.nn.relu(x)

        # last layer: output corresponds to predicted mean and std 
        x = hk.Linear(output_dim)(x) # [B, 2]

        # apply softplus to std
        x = x.at[:, 1].set(jax.nn.softplus(x[:, 1]))

        return x

    return forward


def make_flattened_predict_fn(net, params_sample):
    _, unravel_fn = ravel_pytree(params_sample)

    @jax.jit
    def predict_fn(x, params):
        params = unravel_fn(params)
        y_hat, _ = net.apply(params, None, None, x)
        return y_hat
    
    return predict_fn


def make_nn(key, x, layer_dims=[10, 10, 10], output_dim=2):
    # create NN
    net_fn = make_mlp_fn(layer_dims, output_dim)
    net = hk.transform_with_state(net_fn)
    params, _ = net.init(key, x)
    
    # use arrays as params instead of pytrees
    predict_fn = make_flattened_predict_fn(net, params)
    params = ravel_fn(params)

    return predict_fn, params
