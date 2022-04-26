import jax
import jax.numpy as jnp


def autocorr(chains, n_lags=20):
    # estimate mean of each param
    means = chains.mean(axis=[0, 1])

    # single param, single lag
    def acf(y, mean, lag):
        a = y[n_lags:]
        b = jnp.roll(y, lag)[n_lags:]
        cov_ab = ((a - mean) * (b - mean)).mean()
        cov_a = ((a - mean)**2).mean()
        cov_b = ((b - mean)**2).mean()
        cor = cov_ab / jnp.sqrt(cov_a * cov_b)
        return cor

    # vmap over params
    acf = jax.vmap(acf, in_axes=(1, 0, None))

    # vmap over chains
    acf = jax.vmap(acf, in_axes=(0, None, None))

    # for each lag, average autocor across all chains and params
    def step(i, r):
        cor = acf(chains, means, i).mean()
        r = r.at[i].set(cor)
        return r
    r = jnp.ones([n_lags])
    r = jax.lax.fori_loop(1, n_lags, step, r)
    
    return r


def r_hat(y):
    """
    Based on the paper "What Are Bayesian Neural Network Posteriors Really Like? - Appendix B"
    y.shape: [num_chains (M) x num_steps (N) x num_features (B)]
    """
    M, N, B = y.shape

    # compute variances
    b = (N/(M-1)) * ((y.mean(axis=1, keepdims=True) - y.mean(axis=(0, 1), keepdims=True))**2).sum(axis=(0, 1))
    w = 1/(M*(N-1)) * ((y - y.mean(axis=1, keepdims=True))**2).sum(axis=(0, 1))
    r_hat = jnp.sqrt((((N-1)/N)*w + (1/N)*b)/w)
    
    return r_hat


def r_hat(chains, f):
    """
    Based on the paper "What Are Bayesian Neural Network Posteriors Really Like? - Appendix B"
    chains: [num_chains x num_steps x num_params]
    f: f(chain_node) -> [B]: transforms params to a 1D quantity of interest, eg predictive distribution
    """
    
    # compute function of interest for each chain node
    # - y.shape: [num_chains (M) x num_steps (N) x num_features (B)]
    y = jax.vmap(jax.vmap(f))(chains)
    M, N, B = y.shape

    # compute variances
    b = (N/(M-1)) * ((y.mean(axis=1, keepdims=True) - y.mean(axis=[0, 1], keepdims=True))**2).sum(axis=[0, 1])
    w = 1/(M*(N-1)) * ((y - y.mean(axis=1, keepdims=True))**2).sum(axis=[0, 1])
    r_hat = jnp.sqrt((((N-1)/N)*w + (1/N)*b)/w)
    
    return r_hat
