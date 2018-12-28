import mxnet as mx
import mxnet.ndarray as nd
from mxnet.gluon import nn

import numpy as np
from mxnet.gluon.loss import SigmoidBinaryCrossEntropyLoss

from common import check_gradient
from vae.core import VAELossKLTerm, VAE, VAELoss


def test_vae_loss_kl_term():
    ctx = mx.cpu()
    mu = nd.array([[1., 2.],
                   [0., 1.],
                   [2., 3.]], ctx=ctx)

    sd = nd.array([[1., 0.5],
                   [0.5, 0.5],
                   [2., 1.]], ctx=ctx)

    log_sd = nd.log(sd)

    q = nd.concat(mu, log_sd, dim=1)

    vae_loss_kl_term = VAELossKLTerm(latent_dim=2)

    val = vae_loss_kl_term(q)

    expected = np.array([2.8181471805599454, 1.1362943611198906, 7.306852819440055])

    assert val.shape == (3,)
    assert np.allclose(expected, val.asnumpy())


def test_gradient():
    ctx = mx.cpu()
    latent_dim = 2
    input_dim = 4
    batch_size = 2
    # build the network
    enc_nn = nn.HybridSequential()
    enc_nn.add(nn.Dense(units=3, activation='relu'))
    enc_nn.add(nn.Dense(units=latent_dim * 2))

    dec_nn = nn.HybridSequential()
    dec_nn.add(nn.Dense(units=3, activation='relu'))
    dec_nn.add(nn.Dense(units=input_dim))

    vae_nn = VAE(enc_nn, dec_nn, batch_size, latent_dim)
    model_params = vae_nn.collect_params()
    model_params.initialize(init=mx.init.Uniform(1.0), ctx=ctx)  # don't initialize to small weights

    # loss function
    loss_fn = VAELoss(SigmoidBinaryCrossEntropyLoss(from_sigmoid=False, batch_axis=0), input_dim, latent_dim)

    def fwd(x):
        y, q = vae_nn(x)
        return nd.sum(loss_fn(x, q, y))

    batch_x = mx.nd.random_uniform(shape=(batch_size, input_dim))

    fwd(batch_x)  # the following check fails for the first parameter if fwd is not called at least once before it.
    for p in model_params.values():
        assert check_gradient(fwd, [batch_x], p)

