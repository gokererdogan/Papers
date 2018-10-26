import mxnet as mx
import mxnet.ndarray as nd
import numpy as np

from convdraw.core import ConvDrawLossKLTerm


def test_convdraw_loss_kl_term_1():
    ctx = mx.cpu()
    mu_q = nd.array([[1., 2.],
                     [0., 1.],
                     [2., 3.]], ctx=ctx)

    sd_q = nd.array([[1., 0.5],
                     [0.5, 0.5],
                     [2., 1.]], ctx=ctx)

    mu_p = nd.array([[0., 0.],
                     [0., 0.],
                     [0., 0.]], ctx=ctx)

    sd_p = nd.array([[1., 1.],
                     [1., 1.],
                     [1., 1.]], ctx=ctx)

    log_sd_q = nd.log(sd_q)
    log_sd_p = nd.log(sd_p)

    q = nd.concat(mu_q, log_sd_q, dim=1)
    p = nd.concat(mu_p, log_sd_p, dim=1)

    convdraw_loss_kl_term = ConvDrawLossKLTerm(latent_dim=2)

    val = convdraw_loss_kl_term(q, p)

    expected = np.array([2.8181471805599454, 1.1362943611198906, 7.306852819440055])

    assert val.shape == (3,)
    assert np.allclose(expected, val.asnumpy())


def test_convdraw_loss_kl_term_2():
    ctx = mx.cpu()
    mu_q = nd.array([[1., 2.],
                     [0., 1.],
                     [2., 3.]], ctx=ctx)

    sd_q = nd.array([[1., 0.5],
                     [0.5, 0.5],
                     [2., 1.]], ctx=ctx)

    mu_p = nd.array([[1., 1.],
                     [-0.5, 0.7],
                     [1., 2.3]], ctx=ctx)

    sd_p = nd.array([[0.7, 0.5],
                     [1., 0.3],
                     [1.5, 1.1]], ctx=ctx)

    log_sd_q = nd.log(sd_q)
    log_sd_p = nd.log(sd_p)

    q = nd.concat(mu_q, log_sd_q, dim=1)
    p = nd.concat(mu_p, log_sd_p, dim=1)

    convdraw_loss_kl_term = ConvDrawLossKLTerm(latent_dim=2)

    val = convdraw_loss_kl_term(q, p)

    expected = np.array([2.163733219326574, 1.3212104456828437, 0.5344416978024983])

    assert val.shape == (3,)
    assert np.allclose(expected, val.asnumpy())


def test_convdraw_loss():
    # TODO: implement
    assert False
