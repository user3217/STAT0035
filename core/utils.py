import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten


def ifelse(cond, val_true, val_false):
    return jax.lax.cond(cond, lambda x: x[0], lambda x: x[1], (val_true, val_false))


def ravel_pytree_(pytree):
    """Ravels a pytree like `jax.flatten_util.ravel_pytree`
    but doesn't return a function for unraveling."""
    leaves, treedef = tree_flatten(pytree)
    flat = jnp.concatenate([jnp.ravel(x) for x in leaves])
    return flat


def sample_from_chain(predict_fn, key, x, chain, n_samples=10_000):
    n_samples_per_step = n_samples // len(chain)
    
    # generate samples
    def sample_from_node(key, params):
        y_hat = predict_fn(x, params)
        mean = y_hat[:, 0]
        std = y_hat[:, 1]
        samples = mean[:, None] + std[:, None] * jax.random.normal(key, [len(x), n_samples_per_step])
        return samples
    keys = jax.random.split(key, len(chain))
    samples = jax.vmap(sample_from_node)(keys, chain) # [len(chain), len(x), n_samples_per_step]
    samples = samples.transpose([1, 0, 2]).reshape([len(x), n_samples])
    
    return samples