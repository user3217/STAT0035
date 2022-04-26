import jax
import jax.numpy as jnp
import numpy as onp
import pandas as pd
from . import models


def load_1d_regression_dataset():
    
    # define true function
    def f(x):
        return x[:, 0]/2 + x[:, 0]**2- x[:, 0]**4/4

    # sample true function
    x_true = jnp.linspace(-2, 2, 100)[:, None]
    y_true = f(x_true)

    # observed data
    step = 0.05
    x_train = jnp.concatenate([jnp.arange(-1.1, -0.5, step), jnp.arange(0.5, 1.4, step)])[:, None]
    x_test = jnp.linspace(-2, 2, 100)[:, None]
    y_train = f(x_train)
    y_test = f(x_test)

    # add noise
    sigma_obs = 0.1
    key = jax.random.PRNGKey(0)
    train_noise_key, test_noise_key = jax.random.split(key, 2)
    y_train += sigma_obs * jax.random.normal(train_noise_key, y_train.shape)
    y_test += sigma_obs * jax.random.normal(test_noise_key, y_test.shape)

    return (x_true, y_true), (x_train, y_train), (x_test, y_test)


def load_naval_dataset(ds_path, key, train_fraction=0.9):
    """
    load UCI naval regression dataset
    - x.shape: [11934, 14]
    - y.shape: [11934]
    - ported from: 
      - https://github.com/google-research/google-research/blob/b24af2d59ea3d82799a1a773c896a1c4911b148f/bnn_hmc/utils/data_utils.py#L174
      - https://github.com/wjmaddox/drbayes/blob/0efbd081b7ccecdb2fea8e949ad81065c26faa54/experiments/uci_exps/bayesian_benchmarks/data.py#L157
    """
    # read raw dataset
    data = pd.read_fwf(ds_path, header=None).values
    data = data.astype(onp.float32)

    # only consider the first output
    x = data[:, :-2]
    y = data[:, -2]#.reshape(-1, 1)

    # remove columns 8 and 11, they have have std=0
    x = onp.delete(x, [8, 11], axis=1)

    # shuffle dataset
    indices = jax.random.permutation(key, len(x))
    indices = onp.asarray(indices)
    x, y = x[indices], y[indices]

    # train / test split
    n_train = int(train_fraction * len(x))
    x_train, y_train = x[:n_train], y[:n_train]
    x_test, y_test = x[n_train:], y[n_train:]

    def normalize_with_stats(arr, arr_mean=None, arr_std=None):
        return (arr - arr_mean) / arr_std

    def normalize(arr):
        eps = 1e-6
        arr_mean = arr.mean(axis=0, keepdims=True)
        arr_std = arr.std(axis=0, keepdims=True) + eps
        return normalize_with_stats(arr, arr_mean, arr_std), arr_mean, arr_std

    # normalize
    x_train, x_mean, x_std = normalize(x_train)
    y_train, y_mean, y_std = normalize(y_train)
    x_test = normalize_with_stats(x_test, x_mean, x_std)
    y_test = normalize_with_stats(y_test, y_mean, y_std)

    return (x_train, y_train), (x_test, y_test)

