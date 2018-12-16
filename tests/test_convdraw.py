import mxnet as mx
import mxnet.ndarray as nd
from mxnet.gluon import nn
import numpy as np
import pytest
from mxnet.gluon.loss import SigmoidBinaryCrossEntropyLoss

from common import check_gradient
from convdraw.core import ConvDRAWLossKLTerm, ConvDRAWLoss, ConvDRAW


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

    convdraw_loss_kl_term = ConvDRAWLossKLTerm(latent_dim=2)

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

    convdraw_loss_kl_term = ConvDRAWLossKLTerm(latent_dim=2)

    val = convdraw_loss_kl_term(q, p)

    expected = np.array([2.163733219326574, 1.3212104456828437, 0.5344416978024983])

    assert val.shape == (3,)
    assert np.allclose(expected, val.asnumpy())


def test_convdraw_init():
    with pytest.raises(AssertionError):
        ConvDRAWLoss(None, 5, (3, 5, 5), 2.0)  # input cost scale out of bounds


def test_convdraw_loss():
    ctx = mx.cpu()
    # steps x batch x latent (2 x 3 x 2)
    mu_q = nd.array([[[1., 2.], [0., 1.], [2., 3.]], [[1.5, 0.4], [1.0, 0.7], [1.2, 0.8]]], ctx=ctx)
    sd_q = nd.array([[[1., 0.5], [0.5, 0.5], [2., 1.]], [[0.4, 1.], [0.8, 0.8], [1.5, 2.]]], ctx=ctx)

    mu_p = nd.array([[[1., 1.], [-0.5, 0.7], [1., 2.3]], [[0.5, 1.2], [1.5, 0.8], [1.0, 0.5]]], ctx=ctx)
    sd_p = nd.array([[[0.7, 0.5], [1., 0.3], [1.5, 1.1]], [[0.2, 0.4], [0.6, 0.6], [1.0, 0.3]]], ctx=ctx)

    log_sd_q = nd.log(sd_q)
    log_sd_p = nd.log(sd_p)

    q = nd.concat(mu_q, log_sd_q, dim=2)
    p = nd.concat(mu_p, log_sd_p, dim=2)

    convdraw_loss = ConvDRAWLoss(fit_loss=lambda y, x: 1.0, input_dim=4, latent_shape=(2, 3, 3), input_cost_scale=0.5)

    mock_x = nd.zeros((3, 2, 3), ctx=ctx)
    mock_y = nd.zeros((3, 2, 3), ctx=ctx)
    val = convdraw_loss(mock_x, q, p, mock_y)

    expected_kl = (np.array([2.163733219326574, 1.3212104456828437, 0.5344416978024983]) +
                   np.array([17.015562057495117, 0.5635244846343994, 20.56463623046875]))
    expected_fit = 1.0 * 4 * 0.5

    assert val.shape == (3,)
    assert np.allclose(expected_fit + expected_kl, val.asnumpy())


def test_gradient():
    ctx = mx.cpu()
    num_latent_maps = 1
    input_shape = (1, 2, 2)
    input_dim = 4
    batch_size = 2
    # build the network
    enc_nn = nn.HybridSequential()
    enc_nn.add(nn.Conv2D(channels=2, kernel_size=(1, 1), activation='relu', bias_initializer=mx.init.Uniform(1.0)))

    dec_nn = nn.HybridSequential()
    dec_nn.add(nn.Conv2DTranspose(channels=1, kernel_size=(1, 1), bias_initializer=mx.init.Uniform(1.0)))

    conv_draw_nn = ConvDRAW(enc_nn, dec_nn, num_steps=2, batch_size=batch_size, input_shape=input_shape,
                            num_latent_maps=num_latent_maps, encoder_output_shape=(2, 2, 2), rnn_hidden_channels=1,
                            kernel_size=(1, 1), ctx=ctx)
    model_params = conv_draw_nn.collect_params()
    mx.random.seed(np.random.randint(1000000))
    model_params.initialize(init=mx.init.Uniform(1.0), ctx=ctx)  # don't initialize to small weights

    # loss function
    loss_fn = ConvDRAWLoss(SigmoidBinaryCrossEntropyLoss(from_sigmoid=False, batch_axis=0), input_dim, (1, 2, 2))

    def fwd(x):
        y, q, p = conv_draw_nn(x)
        return nd.sum(loss_fn(x, q, p, y))

    batch_x = mx.nd.random_uniform(shape=(batch_size, *input_shape))

    fwd(batch_x)  # the following check fails for the first parameter is fwd is not called at least once before it.
    for p in model_params.values():
        assert check_gradient(fwd, [batch_x], p)
