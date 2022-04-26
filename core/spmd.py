import jax
import jax.numpy as jnp
from functools import partial


def split(x, n_batches):
    # adds a batch dimension along the first axis
    batch_size = len(x) // n_batches
    x_batched = x[:(n_batches*batch_size)].reshape([n_batches, batch_size, *x.shape[1:]])
    return x_batched


def spmd(f):
    # wrapper to pmap an MCMC sampler 'f' in SPMD fashion
    # f: (log_posterior_fn, params, *args) -> (chain, *logs)

    def out_fn(x, y, params, log_likelihood_fn, log_prior_fn, n_dev=1, *args):
        # batch x, y, params
        x_batched = split(x, n_dev)
        y_batched = split(y, n_dev)
        params_batched = jnp.repeat(params[None], n_dev, axis=0)
        
        # run each batch on a separate device
        @partial(jax.pmap, axis_name='batch')
        def g(x, y, params):
            log_posterior_fn = make_log_posterior_fn(x, y, log_likelihood_fn, log_prior_fn)
            return f(log_posterior_fn, params, *args)
        out_batched = g(x_batched, y_batched, params_batched)
        
        # check that each chain is the same, then return just the first output
        assert jnp.allclose(out_batched[0][0], out_batched[0][1])
        out_single = [out[0] for out in out_batched]
        return out_single
    
    return out_fn


def make_log_posterior_fn(x, y, log_likelihood_fn, log_prior_fn):
    # same as distributions.log_posterior_fn but works in SPMD fashion
    # output values and gradients are synchronized across all devices
    
    @partial(sync_grad_of_psum, axis_name='batch')
    def log_posterior_fn(params):
        log_likelihood = log_likelihood_fn(params, x, y)
        log_likelihood = jax.lax.psum(log_likelihood, axis_name='batch')
        log_prior = log_prior_fn(params)
        return log_likelihood + log_prior

    return log_posterior_fn


def sync_grad_of_psum(f, axis_name):
    """
    If f computes a psum, grad(f) will not be synced across shards
    (hence it will be incorrect). This wrapper modifies f's jvp to fix that.
    - explanation: https://github.com/google/jax/issues/3970
    - implementation details: https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
    """
    def jvp_fn(primals, tangents):
        tangents = jax.lax.pmean(tangents, axis_name=axis_name)
        primals_out, tangents_out = jax.jvp(f, primals, tangents)
        return primals_out, tangents_out
    g = jax.custom_jvp(f)
    g.defjvp(jvp_fn)
    return g
