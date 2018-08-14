"""
An MxNet implementation of the recurrent variational autoencoder described in
    K. Gregor, I. Danihelka, A. Graves, D. J. Rezende, and D. Wierstra,
    “DRAW: A Recurrent Neural Network For Image Generation,”
    arXiv:1502.04623 [cs], Feb. 2015.

13 Aug 2018
goker erdogan
https://github.com/gokererdogan
"""
from typing import Optional

import mxnet as mx
import mxnet.ndarray as nd
from mxnet import autograd
from mxnet.gluon.nn import Dense, HybridBlock
from mxnet.gluon.rnn import LSTMCell

import numpy as np

from common import get_mnist


class DRAW(HybridBlock):
    def __init__(self, num_steps: int, batch_size: int, num_read_units: int, num_rnn_units: int, input_dim: int,
                 latent_dim: int, prefix: Optional[str] = None):
        super().__init__(prefix=prefix)

        self._batch_size = batch_size
        self._num_steps = num_steps
        self._num_read_units = num_read_units
        self._num_rnn_units = num_rnn_units
        self._input_dim = input_dim
        self._latent_dim = latent_dim

        with self.name_scope():
            self._read_layer = Dense(units=self._num_read_units)
            self._enc_rnn = LSTMCell(hidden_size=self._num_rnn_units)
            self._latent_layer = Dense(units=self._latent_dim * 2)  # mean and sd
            self._dec_rnn = LSTMCell(hidden_size=self._num_rnn_units)
            self._write_layer = Dense(units=self._input_dim)

            self._canvas_init = self.params.get('canvas', init=mx.init.Uniform(), shape=(1, self._input_dim,))

            self._enc_rnn_h_init = self.params.get('enc_rnn_h_init', init=mx.init.Uniform(),
                                                   shape=(1, self._num_rnn_units))
            self._enc_rnn_c_init = self.params.get('enc_rnn_c_init', init=mx.init.Uniform(),
                                                   shape=(1, self._num_rnn_units))

            self._dec_rnn_h_init = self.params.get('dec_rnn_h_init', init=mx.init.Uniform(),
                                                   shape=(1, self._num_rnn_units))
            self._dec_rnn_c_init = self.params.get('dec_rnn_c_init', init=mx.init.Uniform(),
                                                   shape=(1, self._num_rnn_units))

    def hybrid_forward(self, F, x, *args, **kwargs):
        with x.context:
            canvas = F.broadcast_to(self._canvas_init.data(), (self._batch_size, 0))
            h_enc = F.broadcast_to(self._enc_rnn_h_init.data(), (self._batch_size, 0))
            c_enc = F.broadcast_to(self._enc_rnn_c_init.data(), (self._batch_size, 0))
            h_dec = F.broadcast_to(self._dec_rnn_h_init.data(), (self._batch_size, 0))
            c_dec = F.broadcast_to(self._dec_rnn_c_init.data(), (self._batch_size, 0))

            zs = []

            for i in range(self._num_steps):
                err = x - F.sigmoid(canvas)
                r = self._read_layer(F.concat(x, err))
                _, (h_enc, c_enc) = self._enc_rnn(F.concat(r, h_dec, c_dec), [h_enc, c_enc])

                q = self._latent_layer(h_enc)
                z = q[:, 0:self._latent_dim] + \
                    F.random.normal(shape=(self._batch_size, self._latent_dim)) * F.exp(q[:, self._latent_dim:])
                zs.append(z)

                _, (h_dec, c_dec) = self._dec_rnn(z, [h_dec, c_dec])
                w = self._write_layer(h_dec)
                canvas = canvas + w

        return F.sigmoid(canvas), zs


if __name__ == "__main__":
    ctx = mx.cpu()
    input_shape = (28, 28)
    input_dim = input_shape[0] * input_shape[1]
    read_dim = 512
    latent_dim = 100
    rnn_dim = 256
    batch_size = 32
    num_steps = 8

    train_set, test_set = get_mnist(batch_size, input_shape)

    draw_nn = DRAW(num_steps=num_steps, batch_size=batch_size, num_read_units=read_dim, num_rnn_units=rnn_dim,
                   input_dim=input_dim, latent_dim=latent_dim)

    draw_nn.collect_params().initialize(ctx=ctx)

    batch = train_set.__next__()
    xhat, zs = draw_nn(batch.data[0])

    pass
