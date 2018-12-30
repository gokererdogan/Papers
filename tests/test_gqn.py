import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
from mxnet.gluon import nn, HybridBlock

from common import check_gradient
from gqn.core import RepresentationNetworkTower, NormalFitLoss, GenerativeQueryNetwork, GQNLoss


def test_normal_fit_loss():
    batch_size = 3
    input_shape = (2, 10, 10)
    std = 0.5
    ctx = mx.cpu()

    loss_fn = NormalFitLoss(std=std)

    pred = (np.ones((*input_shape, batch_size)) * np.array([0.4, 0.2, 0.9])).transpose((3, 0, 1, 2))
    true = (np.ones((*input_shape, batch_size)) * np.array([0.0, 0.1, 0.2])).transpose((3, 0, 1, 2))

    loss = loss_fn(nd.array(pred, ctx=ctx), nd.array(true, ctx=ctx))
    assert loss.shape == (batch_size,)
    assert np.allclose(loss.asnumpy(), np.array([109.15827052894554/200, 49.15827052894551/200, 241.1582705289455/200]))


def test_representation_network():
    batch_size = 3
    context_size = 2
    num_camera_params = 7
    rep_nn = RepresentationNetworkTower(batch_size=3, num_camera_params=num_camera_params)
    rep_nn.initialize(ctx=mx.cpu(), init=mx.init.Uniform(1.0))

    frames = nd.random.uniform(shape=(batch_size, context_size, 3, 64, 64))
    cameras = nd.random.uniform(shape=(batch_size, context_size, num_camera_params))

    out = rep_nn(frames, cameras)

    assert out.shape == (batch_size, 256, 16, 16)

    # check that each scene is processed separately
    rep_nn._batch_size = 1
    outs = [rep_nn(f, c) for f, c in zip(frames, cameras)]
    for i, out_i in enumerate(outs):
        assert np.allclose(out_i.asnumpy(), out[i].asnumpy())


class _MockRepresentationNetwork(HybridBlock):
    def __init__(self, input_shape, batch_size, num_camera_params):
        super().__init__()

        self._num_camera_params = num_camera_params
        self._batch_size = batch_size
        self._input_shape = input_shape

        with self.name_scope():
            self._conv = nn.Conv2D(channels=1, kernel_size=(1, 1), activation='relu',
                                   bias_initializer=mx.init.Uniform(1.0))

    def hybrid_forward(self, F, x, *args, **kwargs):
        # inputs: x: frames (BxKxCxHxW), args[0]: cameras (viewpoints) (BxKxF)
        # K is the context dimension
        v = args[0]

        # get rid of context dimension so we can feed the images to conv layers
        x = F.reshape(x, (-1, *self._input_shape))
        v = F.reshape(v, (-1, self._num_camera_params))

        # concat camera information
        v = F.broadcast_to(F.expand_dims(F.expand_dims(v, axis=-1), axis=-1), (0, 0, *self._input_shape[1:]))
        y = F.concat(x, v, dim=1)

        y = self._conv(y)

        # sum over context
        r = F.sum(F.reshape(y, (self._batch_size, -1, 1, *self._input_shape[1:])), axis=1)
        return r


def test_gradient():
    ctx = mx.cpu()
    num_latent_maps = 1
    input_shape = (1, 2, 2)
    input_dim = 4
    batch_size = 2
    context_size = 2
    num_camera_params = 2

    # build the network
    rep_nn = _MockRepresentationNetwork(input_shape, batch_size, num_camera_params)

    enc_nn = nn.HybridSequential()
    enc_nn.add(nn.Conv2D(channels=2, kernel_size=(1, 1), activation='relu', bias_initializer=mx.init.Uniform(1.0)))

    dec_nn = nn.HybridSequential()
    dec_nn.add(nn.Conv2DTranspose(channels=1, kernel_size=(1, 1), bias_initializer=mx.init.Uniform(1.0)))

    gqn_nn = GenerativeQueryNetwork(rep_nn, enc_nn, dec_nn, num_steps=2, batch_size=batch_size, input_shape=input_shape,
                                    num_latent_maps=num_latent_maps, num_camera_params=num_camera_params,
                                    downsample_output_shape=(2, 2, 2), representation_shape=(1, 2, 2),
                                    upsample_output_shape=(1, 2, 2),
                                    rnn_hidden_channels=1, kernel_size=(1, 1), ctx=ctx)
    model_params = gqn_nn.collect_params()
    mx.random.seed(np.random.randint(1000000))
    model_params.initialize(init=mx.init.Uniform(1.0), ctx=ctx)  # don't initialize to small weights

    # loss function
    loss_fn = GQNLoss(NormalFitLoss(std=0.5), input_dim, (1, 2, 2))

    def fwd(x_q, v_q, x_context, v_context):
        y, q, p = gqn_nn(x_q, v_q, x_context, v_context)
        return nd.sum(loss_fn(x_q, q, p, y))

    batch_x_q = mx.nd.random_uniform(shape=(batch_size, *input_shape))
    batch_v_q = mx.nd.random_uniform(shape=(batch_size, num_camera_params))
    batch_x_context = mx.nd.random_uniform(shape=(batch_size, context_size, *input_shape))
    batch_v_context = mx.nd.random_uniform(shape=(batch_size, context_size, num_camera_params))

    # the gradient check fails for the first parameter if fwd is not called at least once before it.
    fwd(batch_x_q, batch_v_q, batch_x_context, batch_v_context)
    for p in model_params.values():
        assert check_gradient(fwd, [batch_x_q, batch_v_q, batch_x_context, batch_v_context], p)
