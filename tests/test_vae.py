import mxnet as mx
import mxnet.ndarray as nd

import numpy as np

from vae.core import VAELossKLTerm


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

    vae_loss_kl_term = VAELossKLTerm(latent_dim=2, batch_axis=0)

    val = vae_loss_kl_term(q)

    expected = np.array([2.8181471805599454, 1.1362943611198906, 7.306852819440055])

    assert np.allclose(expected, val.asnumpy())
