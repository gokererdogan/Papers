import mxnet as mx
import mxnet.ndarray as nd

import numpy as np
from mxnet.gluon.loss import SigmoidBinaryCrossEntropyLoss

from common import check_gradient
from draw.core import DRAWLoss, NoAttentionRead, NoAttentionWrite, DRAW


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


def test_gradient():
    ctx = mx.cpu()
    input_dim = 3
    num_steps = 10
    latent_dim = 2
    batch_size = 3
    num_recurrent_units = 3

    # build the network
    read_nn = NoAttentionRead()
    write_nn = NoAttentionWrite(units=input_dim)

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

    # NOTE this occasionally fails for initial canvas parameter. I don't know exactly why but the values are still
    # close.
    for p in model_params.values():
        assert check_gradient(fwd, [batch_x], p)

