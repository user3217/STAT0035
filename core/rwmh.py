import jax
import jax.numpy as jnp
from .utils import ifelse


def rwmh_sampler(log_prob_fn, params, key, n_steps, n_blind_steps, step_size):
    
    # define a step that doesn't keep history
    def step_without_history(i, args):
        params, log_prob, total_accept_prob, key = args
        key, normal_key, uniform_key = jax.random.split(key, 3)
        
        # propose new parameters
        step = step_size*jax.random.normal(normal_key, params.shape)
        params_new = params + step
        
        # decide whether to accept new position
        log_prob_new = log_prob_fn(params_new)
        log_accept_prob = log_prob_new - log_prob
        accept_prob = jnp.minimum(1, jnp.exp(log_accept_prob))
        total_accept_prob += accept_prob
        accept = jax.random.uniform(uniform_key) < accept_prob
        params = ifelse(accept, params_new, params)
        log_prob = ifelse(accept, log_prob_new, log_prob)
        
        return params, log_prob, total_accept_prob, key
    
    # define a step that keeps history
    def step_with_history(i, args):
        params, params_history, total_accept_prob, key = args
        
        # do 'n_blind_steps', without keeping history
        log_prob = log_prob_fn(params)
        params, log_prob, inner_total_accept_prob, key = jax.lax.fori_loop(0, n_blind_steps, step_without_history, (params, log_prob, 0, key))
        total_accept_prob += inner_total_accept_prob / n_blind_steps
        
        # store params
        params_history = params_history.at[i].set(params)
              
        return params, params_history, total_accept_prob, key
    
    # do 'n_steps'
    params_history = jnp.zeros([n_steps, len(params)])
    _, params_history, total_accept_prob, key = jax.lax.fori_loop(0, n_steps, step_with_history, (params, params_history, 0, key))
    avg_accept_prob = total_accept_prob/n_steps

    return params_history, avg_accept_prob
