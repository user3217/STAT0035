import jax
import jax.numpy as jnp
from .utils import ifelse


"""
Notation:
- s: model parameters
- v: momentum
- u: slice parameter

References:
[1]: "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo"
[2]: "Composable Effects for Flexible and Accelerated Probabilistic Programming in NumPyro"
"""


def is_power_of_two(n):
    # https://stackoverflow.com/a/57025941/6495494
    return ((n & (n-1)) == 0) & (n != 0)


def bit_count(n):
    # computes the number of non-zero bits
    # https://github.com/pyro-ppl/numpyro/blob/38b0d4879e6300a17fe46a1ec434b8620dbdfe16/numpyro/infer/hmc_util.py#L941

    def while_cond(args):
        n, count = args
        # stop when n = 0, ie all bits are zero
        return n > 0

    def step(args):
        n, count = args

        # if the last binary digit of n is 1, increment count by 1
        count += n & 1
        
        # roll the bits of n
        n >>= 1

        return n, count
    
    _, count = jax.lax.while_loop(while_cond, step, (n, 0))
    return count


def num_trailing_bits(n):
    # computes the number of contiguous last non-zero bits
    # https://github.com/pyro-ppl/numpyro/blob/38b0d4879e6300a17fe46a1ec434b8620dbdfe16/numpyro/infer/hmc_util.py#L941

    def while_cond(args):
        n, count = args
        # stop when the last binary digit is zero => continue if it's one
        return (n & 1) == 1

    def step(args):
        n, count = args

        # if the last binary digit of n is 1, increment count by 1
        count += n & 1
        
        # roll the bits of n
        n >>= 1

        return n, count
    
    _, count = jax.lax.while_loop(while_cond, step, (n, 0))
    return count


def leapfrog_step(s, v, log_prob_fn, step_size):
    """
    Approximates Hamiltonian dynamics using the leapfrog algorithm.
    https://github.com/google-research/google-research/blob/eb1aae0a0e80b94436b52b61304e534d226c9017/bnn_hmc/core/hmc.py#L37
    """
        
    # update momentum
    grad = jax.grad(log_prob_fn)(s)
    v = jax.tree_multimap(lambda v, g: v + 0.5 * step_size * g, v, grad)

    # update params
    s = jax.tree_multimap(lambda s, v: s + v * step_size, s, v)

    # update momentum
    grad = jax.grad(log_prob_fn)(s)
    v = jax.tree_multimap(lambda v, g: v + 0.5 * step_size * g, v, grad)
    
    return s, v


def kinetic_energy_fn(v):
    return 0.5*sum([jnp.sum(m**2) for m in jax.tree_leaves(v)])


def approx_error_too_large(s, v, log_u, log_prob_fn, max_error=1000):
    # TODO: what's a reasonable value of max_error?
    """Ref. [1], eq. (8), adapted from Algorithm 3."""
    too_large = log_u > max_error + log_prob_fn(s) - kinetic_energy_fn(v)
    return too_large


def satisfies_detailed_balance(s, v, log_u, log_prob_fn):
    """Ref [1], C.3"""
    return log_u < log_prob_fn(s) - kinetic_energy_fn(v)


def is_u_turn(s_fwd, v_fwd, s_bwd, v_bwd, direction):
    """Ref. [1], eq. (9)."""

    # check for a U-turn in both directions
    stop_fwd = direction*jnp.dot((s_fwd - s_bwd), v_bwd) < 0
    stop_bwd = direction*jnp.dot((s_fwd - s_bwd), v_fwd) < 0

    return stop_fwd | stop_bwd


def check_uturns(s_fwd, v_fwd, i, checkpoints, direction):

    # find which checkpoints need to be used [2], Algorithm 2
    num_checks = num_trailing_bits(i)
    last_idx = bit_count(i) - 1
    first_idx = last_idx - num_checks + 1

    # loop through each relevant checkpoint
    def step(j, stop):
        s_bwd = checkpoints[j, 0]
        v_bwd = checkpoints[j, 1]
        stop |= is_u_turn(s_fwd, v_fwd, s_bwd, v_bwd, direction)
        return stop
    stop = jax.lax.fori_loop(first_idx, last_idx+1, step, False)

    return stop


def save_checkpoint(s, v, i, checkpoints):
    j = bit_count(i)
    checkpoints = checkpoints.at[j, 0].set(s)
    checkpoints = checkpoints.at[j, 1].set(v)
    return checkpoints


def make_nuts_step(log_prob_fn, step_size, max_depth):

    def nuts_step(s, v, key):
        """
        Runs one iteration of NUTS and proposes a new (params, momentum) pair.
        Uses an iterative implementation of the NUTS algorithm [2].
        """
        key, slice_key, loop_key = jax.random.split(key, 3)

        # create a 'checkpoints' vector to store intermediate nodes in the chain
        # - in the second dimension, index 0 stores s, index 1 stores v
        checkpoints = jnp.zeros([max_depth, 2, len(s)])

        # checkpoint current state
        checkpoints = checkpoints.at[0, 0].set(s)
        checkpoints = checkpoints.at[0, 1].set(v)

        # store the leftmost and rightmost state (currently the same state)
        s_fwd, v_fwd = s, v
        s_bwd, v_bwd = s, v

        # create a placeholder to store the randomly-sampled output state
        s_rand, v_rand = s, v

        # sample slice variable
        # - to avoid underflow, log(u) has to be used instead of u
        # - here's a derivation of the distribution of log(u)
        #   - u ~ Unif[0, a]
        #   - let x = u/a => x ~ Unif[0, 1]
        #   - let y = -log(x) => y ~ Exp(1)
        #   - log(u) = log(a*x) = log(a) - (-log(x)) = log(a) - y
        log_u = (log_prob_fn(s) - kinetic_energy_fn(v)) - jax.random.exponential(key)

        # 'stop' will store the reason NUTS has terminated
        # - 0: ran out of steps
        # - 1: u-turn
        # - 2: approx. error too large
        stop = jnp.zeros([3], dtype=bool)

        # when the tree doubles
        # - the direction has to be resampled and it might change
        # - if the direction changes, the current tree needs to be flipped
        def double_tree(args):
            direction, s_fwd, v_fwd, s_bwd, v_bwd, s_out, v_out, s_rand, v_rand, checkpoints, dir_key = args

            # resample direction
            new_direction = 2*jax.random.bernoulli(dir_key) - 1
            direction_changed = new_direction == direction
            direction = new_direction
        
            # if direction changed, flip the tree
            s_fwd, s_bwd = ifelse(direction_changed, (s_bwd, s_fwd), (s_fwd, s_bwd))
            v_fwd, v_bwd = ifelse(direction_changed, (v_bwd, v_fwd), (v_fwd, v_bwd))

            # update checkpoints
            # - regardless of the current direction, the only relevant
            #   checkpoint from the previous subtree is the outer-most one
            # - this is always denoted 'bwd' in the new tree
            checkpoints = checkpoints.at[0, 0].set(s_bwd)
            checkpoints = checkpoints.at[0, 1].set(v_bwd)

            # update output
            s_out, v_out = s_rand, v_rand

            return direction, s_fwd, v_fwd, s_bwd, v_bwd, s_out, v_out, s_rand, v_rand, checkpoints, dir_key

        # define a single leapgfrog step with all overhead logic (eg storing intermediate state, stopping condition)
        def step(args):
            i, s_out, v_out, s_rand, v_rand, s_fwd, v_fwd, s_bwd, v_bwd, checkpoints, direction, n_valid_samples, stop, key = args
            key, dir_key, error_key, keep_key = jax.random.split(key, 4)

            # if the current step is a power of 2, the tree is about to doble
            tree_has_doubled = is_power_of_two(i)
            args = direction, s_fwd, v_fwd, s_bwd, v_bwd, s_out, v_out, s_rand, v_rand, checkpoints, dir_key
            args = jax.lax.cond(tree_has_doubled, double_tree, lambda x: x, args)
            direction, s_fwd, v_fwd, s_bwd, v_bwd, s_out, v_out, s_rand, v_rand, checkpoints, _ = args

            # leapfrog
            # - if 'direction=-1', will run backwards in time
            s_fwd, v_fwd = leapfrog_step(s_fwd, v_fwd, log_prob_fn, direction*step_size)

            # if i is odd, check for a u-turn
            is_u_turn = ifelse(i % 2 == 1, check_uturns(s_fwd, v_fwd, i, checkpoints, direction), False)
            stop = stop.at[1].set(is_u_turn)

            # update checkpoints
            # - this only needs to be done if i is even, but for simplicity, here it is always done
            checkpoints = save_checkpoint(s_fwd, v_fwd, i, checkpoints)

            # check if leapfrog error is too large
            error_too_large = approx_error_too_large(s, v, log_u, log_prob_fn)
            stop = stop.at[2].set(error_too_large)

            # check if we reached the maximum number of steps
            max_steps_reached = i == (2**max_depth - 1)
            stop = stop.at[0].set(max_steps_reached)

            # update output state
            # - each state that satisfies detailed balance has equal prob. of being chosen
            # - to achieve this, at each step, we replace the current output with prob. 1/n
            #   where n is the total number of samples that satisfy detailed balance
            keep = satisfies_detailed_balance(s_fwd, v_fwd, log_u, log_prob_fn)
            n_valid_samples = ifelse(keep, n_valid_samples+1, n_valid_samples)
            p_update = 1 / n_valid_samples
            do_update = keep & (jax.random.uniform(keep_key) < p_update)
            s_rand = ifelse(do_update, s_fwd, s_rand)
            v_rand = ifelse(do_update, v_fwd, v_rand)

            return i+1, s_out, v_out, s_rand, v_rand, s_fwd, v_fwd, s_bwd, v_bwd, checkpoints, direction, n_valid_samples, stop, key

        # define a stopping condition for the lax while loop
        # - stop is in int (rather than a bool) since jit tracing fails with a bool
        def while_cond(args):
            *_, stop, _ = args
            return stop.sum() == 0

        # leapfrog until a stopping condition is reached
        args = (1, s, v, s, v, s, v, s, v, checkpoints, 1, 1, stop, key)
        # while while_cond(args): args = step(args)
        args = jax.lax.while_loop(while_cond, step, args)
        i, s_out, v_out, *_, n_valid_samples, stop, _ = args

        return s_out, v_out, i, n_valid_samples, stop

    return nuts_step


def nuts_sampler(log_prob_fn, s, key, n_steps, max_depth, step_size):
    """Runs the No-U-Turn Sampler (NUTS) variant of HMC for n_steps."""

    # make a function that will compute one step of the NUTs algorithm
    nuts_step = make_nuts_step(log_prob_fn, step_size, max_depth)

    # make a function that will compute one step of the Metropolis–Hastings algorithm, using NUTS proposals
    def step(i, args):
        s, s_history, total_steps_taken, total_valid_samples, total_stops, key = args
        key, v_key, nuts_key, accept_key = jax.random.split(key, 4)

        # sample momentum
        v = jax.random.normal(v_key, s.shape)

        # propose new params and momentum
        s, v, n_steps_taken, n_valid_samples, stop = nuts_step(s, v, key)

        # store history
        s_history = s_history.at[i].set(s)
        total_steps_taken += n_steps_taken
        total_valid_samples += n_valid_samples
        total_stops += stop
         
        return s, s_history, total_steps_taken, total_valid_samples, total_stops, key
    
    # do 'n_steps'
    # - `params` is a pytree of the current parameters
    # - `params_history_raveled` is the output chain represented as a 2D array
    s_history = jnp.zeros([n_steps, len(s)])
    total_stops = jnp.zeros([3], dtype=jnp.int32)
    args = s, s_history, 0, 0, total_stops, key
    # for i in range(0, n_steps): args = step(i, args)
    args = jax.lax.fori_loop(0, n_steps, step, args)
    _, s_history, total_steps_taken, total_valid_samples, total_stops, key = args

    # print(f'Termination: max_steps={total_stops[0]}, u-turn={total_stops[1]}, max_error={total_stops[2]}.')
    ratio_valid_samples = total_valid_samples/total_steps_taken
    return s_history, ratio_valid_samples, total_stops
