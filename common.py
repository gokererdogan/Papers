from typing import List

import mxnet as mx
from mxnet import autograd

import numpy as np
from mxnet.gluon import Parameter

get_mnist = mx.test_utils.get_mnist_iterator


def check_gradient(forward_fn, fn_params: List[mx.ndarray.NDArray], wrt: Parameter, seed=None, eps=3e-4,
                   tol=1e-2) -> bool:
    """
    Check autograd backward for a given function using finite differencing.

    :param forward_fn: The function to test the gradients of. This function should return a scalar.
    :param fn_params: A list of parameters to call the function.
    :param wrt: The parameter with respect to which we take the gradient.
    :param seed: Random seed for mxnet and numpy. Note that the forward function might be stochastic. We reinitialize
        the seed to the same number before every forward function call.
    :param eps: Epsilon used in finite differencing. The default value is taken from theano's verify_grad function.
    :param tol: Absolute and relative tolerance used to check equality. Again, the default value is taken from theano's
        verify_grad function.
    :return: True if check succeeds.
    """
    if seed is None:
        seed = int(np.random.rand() * 1e6)

    # calculate gradient with autograd
    mx.random.seed(seed)
    np.random.seed(seed)
    with autograd.record():
        out = forward_fn(*fn_params)

    autograd.backward(out)
    ag_grad = wrt.grad().asnumpy()

    # calculate gradient by finite difference
    orig_data = wrt.data().asnumpy()
    fd_grad = np.zeros_like(orig_data)
    for i in range(orig_data.size):
        ix = np.unravel_index(i, orig_data.shape)

        # f(x + h)
        orig_data[ix] += eps
        wrt.set_data(orig_data)
        mx.random.seed(seed)
        np.random.seed(seed)
        out_ph = forward_fn(*fn_params).asscalar()

        # f(x - h)
        orig_data[ix] -= (2*eps)
        wrt.set_data(orig_data)
        mx.random.seed(seed)
        np.random.seed(seed)
        out_mh = forward_fn(*fn_params).asscalar()
        orig_data[ix] += eps  # revert

        # calc gradient
        fd_grad[ix] = (out_ph - out_mh) / (2 * eps)

    return np.allclose(ag_grad, fd_grad, atol=tol, rtol=tol)


