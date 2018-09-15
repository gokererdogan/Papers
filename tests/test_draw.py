import mxnet as mx
import mxnet.ndarray as nd

import numpy as np
from mxnet.gluon.loss import SigmoidBinaryCrossEntropyLoss

from common import check_gradient
from draw.core import DRAWLoss, NoAttentionRead, NoAttentionWrite, DRAW, SelectiveAttentionBase, SelectiveAttentionRead, \
    SelectiveAttentionWrite


def test_draw_loss():
    ctx = mx.cpu()
    # num_steps=2
    mu = nd.zeros((2, 3, 2))  # steps x batch x latent
    mu[0] = nd.array([[1., 2.],
                      [0., 1.],
                      [2., 3.]], ctx=ctx)
    mu[1] = nd.array([[-1., 1.],
                      [-2., 1.5],
                      [2., -1.]], ctx=ctx)

    sd = nd.zeros((2, 3, 2))  # steps x batch x latent
    sd[0] = nd.array([[1., 0.5],
                      [0.5, 0.5],
                      [2., 1.]], ctx=ctx)
    sd[1] = nd.array([[2., 0.2],
                      [1.5, 1.5],
                      [.4, 3.]], ctx=ctx)

    log_sd = nd.log(sd)

    qs = nd.concat(mu, log_sd, dim=2)

    # we only test the kl term and don't care about the fit term.
    draw_loss = DRAWLoss(fit_loss=lambda y, x: 0.0, input_dim=1, latent_dim=2)

    mock_x = nd.ones((3, 1))
    mock_y = nd.ones((3, 1))

    val = draw_loss(mock_x, qs, mock_y)

    expected = (np.array([2.8181471805599454, 1.1362943611198906, 7.306852819440055]) +
                np.array([2.9362907318741547, 3.564069783783671, 5.897678443206045]))

    assert val.shape == (3,)
    assert np.allclose(expected, val.asnumpy())


def test_selective_attention_build_filter():
    # filter 3x3
    # image 4x5
    # batch size 2
    attn_block = SelectiveAttentionBase(filter_size=3, input_shape=(4, 5), batch_size=2)

    attn_params = nd.zeros((2, 5))
    attn_params[1, 0] = -1.  # gx for batch 2
    attn_params[1, 1] = -2.  # gy for batch 2
    attn_params[1, 2] = np.log(2.)  # log var for batch 2
    attn_params[1, 3] = np.log(1/2.)  # log delta for batch 2

    Fx, Fy = attn_block._build_filter(nd, attn_params)

    # 3x4
    expected_Fx_1 = np.array([[-1/8., -9/8., -25/8., -49/8.],
                              [-9/8., -1/8., -1/8., -9/8.],
                              [-49/8., -25/8., -9/8., -1/8.]])
    expected_Fx_1 = np.exp(expected_Fx_1)
    expected_Fx_1 = (expected_Fx_1.T / np.sum(expected_Fx_1, axis=1)).T

    expected_Fx_2 = np.array([[-1., -9/4., -4., -25/4.],
                              [-1/4., -1., -9/4., -4.],
                              [-0., -1/4., -1., -9/4.]])
    expected_Fx_2 = np.exp(expected_Fx_2)
    expected_Fx_2 = (expected_Fx_2.T / np.sum(expected_Fx_2, axis=1)).T

    # 3x5
    expected_Fy_1 = np.array([[-0., -1/2., -2., -9/2., -8.],
                              [-2., -1/2., -0., -1/2., -2.],
                              [-8., -9/2., -2., -1/2., -0.]])
    expected_Fy_1 = np.exp(expected_Fy_1)
    expected_Fy_1 = (expected_Fy_1.T / np.sum(expected_Fy_1, axis=1)).T

    expected_Fy_2 = np.array([[-25/4., -9., -49/4., -16., -81/4.],
                              [-4., -25/4., -9., -49/4., -16.],
                              [-9/4., -4., -25/4., -9., -49/4.]])
    expected_Fy_2 = np.exp(expected_Fy_2)
    expected_Fy_2 = (expected_Fy_2.T / np.sum(expected_Fy_2, axis=1)).T

    assert np.allclose(expected_Fx_1, Fx[0].asnumpy())
    assert np.allclose(expected_Fx_2, Fx[1].asnumpy())
    assert np.allclose(expected_Fy_1, Fy[0].asnumpy())
    assert np.allclose(expected_Fy_2, Fy[1].asnumpy())


def test_selective_attention_read():
    # filter 3x3
    # image 4x5
    # batch size 2
    ctx = mx.cpu()
    attn_block = SelectiveAttentionRead(filter_size=3, input_shape=(4, 5), batch_size=2)
    attn_block.collect_params().initialize(ctx=ctx)

    x = nd.random.uniform(shape=(2, 4, 5), ctx=ctx)
    err = nd.random.uniform(shape=(2, 4, 5), ctx=ctx)
    h_dec = nd.random.normal(shape=(2, 3), ctx=ctx)
    c_dec = nd.random.normal(shape=(2, 3), ctx=ctx)

    attn_params = attn_block._attention_params_layer(nd.concat(h_dec, c_dec, dim=1))
    Fx, Fy = attn_block._build_filter(nd, attn_params)

    read, _ = attn_block(x, err, h_dec, c_dec)

    # calculate expected
    expected_x_1 = (np.dot(np.dot(Fx[0].asnumpy(), x[0].asnumpy()), Fy[0].asnumpy().T) *
                    np.exp(attn_params[0, 4].asscalar()))
    expected_err_1 = (np.dot(np.dot(Fx[0].asnumpy(), err[0].asnumpy()), Fy[0].asnumpy().T) *
                      np.exp(attn_params[0, 4].asscalar()))
    expected_1 = np.concatenate([expected_x_1.flatten(), expected_err_1.flatten()])

    expected_x_2 = (np.dot(np.dot(Fx[1].asnumpy(), x[1].asnumpy()), Fy[1].asnumpy().T) *
                    np.exp(attn_params[1, 4].asscalar()))
    expected_err_2 = (np.dot(np.dot(Fx[1].asnumpy(), err[1].asnumpy()), Fy[1].asnumpy().T) *
                      np.exp(attn_params[1, 4].asscalar()))
    expected_2 = np.concatenate([expected_x_2.flatten(), expected_err_2.flatten()])
    expected = np.stack([expected_1, expected_2], axis=0)

    assert np.allclose(expected, read.asnumpy())


def test_selective_attention_write():
    # filter 3x3
    # image 4x5
    # batch size 2
    ctx = mx.cpu()
    attn_block = SelectiveAttentionWrite(filter_size=3, input_shape=(4, 5), batch_size=2)
    attn_block.collect_params().initialize(ctx=ctx)

    h_dec = nd.random.normal(shape=(2, 3), ctx=ctx)
    c_dec = nd.random.normal(shape=(2, 3), ctx=ctx)

    attn_params = attn_block._attention_params_layer(nd.concat(h_dec, c_dec, dim=1))
    Fx, Fy = attn_block._build_filter(nd, attn_params)

    write, _ = attn_block(h_dec, c_dec)

    # calculate expected
    w = nd.reshape(attn_block._patch_layer(nd.concat(h_dec, c_dec, dim=1)), (-1, 3, 3))
    expected_1 = (np.dot(np.dot(Fx[0].asnumpy().T, w[0].asnumpy()), Fy[0].asnumpy()) /
                    np.exp(attn_params[0, 4].asscalar()))
    expected_2 = (np.dot(np.dot(Fx[1].asnumpy().T, w[1].asnumpy()), Fy[1].asnumpy()) /
                    np.exp(attn_params[1, 4].asscalar()))
    expected = np.stack([expected_1.flatten(), expected_2.flatten()], axis=0)

    assert np.allclose(expected, write.asnumpy())


def test_gradient_wo_attention():
    ctx = mx.cpu()
    input_dim = 3
    num_steps = 10
    latent_dim = 2
    batch_size = 3
    num_recurrent_units = 3

    # build the network
    read_nn = NoAttentionRead()
    write_nn = NoAttentionWrite(input_dim)

    draw_nn = DRAW(read_nn, write_nn, num_steps, batch_size, num_recurrent_units, input_dim,
                   latent_dim)
    model_params = draw_nn.collect_params()
    model_params.initialize(init=mx.init.Uniform(1.0), ctx=ctx)

    # loss function
    loss_fn = DRAWLoss(SigmoidBinaryCrossEntropyLoss(from_sigmoid=False, batch_axis=0), input_dim, latent_dim)

    def fwd(x):
        y, qs = draw_nn(x)
        return nd.sum(loss_fn(x, qs, y))

    batch_x = mx.nd.random_uniform(shape=(batch_size, input_dim))

    # TODO: investigate why this fails if we remove the following fwd call.
    fwd(batch_x)
    for p in model_params.values():
        assert check_gradient(fwd, [batch_x], p)


def test_gradient_with_attention():
    ctx = mx.cpu()
    input_shape = (2, 2)
    input_dim = 4
    num_steps = 10
    latent_dim = 2
    batch_size = 3
    num_recurrent_units = 3

    # build the network
    read_nn = SelectiveAttentionRead(2, input_shape, batch_size)
    write_nn = SelectiveAttentionWrite(2, input_shape, batch_size)

    draw_nn = DRAW(read_nn, write_nn, num_steps, batch_size, num_recurrent_units, input_dim,
                   latent_dim)
    model_params = draw_nn.collect_params()
    model_params.initialize(init=mx.init.Uniform(1.0), ctx=ctx)

    # loss function
    loss_fn = DRAWLoss(SigmoidBinaryCrossEntropyLoss(from_sigmoid=False, batch_axis=0), input_dim, latent_dim)

    def fwd(x):
        y, qs = draw_nn(x)
        return nd.sum(loss_fn(x, qs, y))

    batch_x = mx.nd.random_uniform(shape=(batch_size, input_dim))

    # TODO: investigate why this fails for the first parameters if fwd is not called once before check gradient is
    # called
    fwd(batch_x)
    for p in model_params.values():
        assert check_gradient(fwd, [batch_x], p)

