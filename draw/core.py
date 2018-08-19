"""
An MxNet implementation of the recurrent variational autoencoder described in
    K. Gregor, I. Danihelka, A. Graves, D. J. Rezende, and D. Wierstra,
    “DRAW: A Recurrent Neural Network For Image Generation,”
    arXiv:1502.04623 [cs], Feb. 2015.

13 Aug 2018
goker erdogan
https://github.com/gokererdogan
"""
from typing import Optional, Tuple

import mxnet as mx
import mxnet.ndarray as nd
from mxnet.gluon.loss import Loss
from mxnet.gluon.nn import Dense, HybridBlock
from mxnet.gluon.rnn import LSTMCell

from vae.core import NormalSamplingBlock, VAELoss


class NoAttentionRead(HybridBlock):
    """
    Read block with no attention.

    Section 3.1 in the paper. This simply concatenates the input image and the error image.
    """
    def __init__(self, prefix: Optional[str] = None):
        super().__init__(prefix=prefix)

    def hybrid_forward(self, F, x, *args, **kwargs):
        err = args[0]
        return F.concat(x, err, dim=1)


class NoAttentionWrite(HybridBlock):
    """
    Read block with no attention.

    Section 3.1 in the paper. This is simply a fully connected linear layer.
    """
    def __init__(self, input_dim: int, prefix: Optional[str] = None):
        """
        :param input_dim: Number of elements in input (e.g., height x width for image input).
        :param prefix: Prefix for block.
        """
        super().__init__(prefix=prefix)

        with self.name_scope():
            self._write_layer = Dense(units=input_dim)

    def hybrid_forward(self, F, h_dec, *args, **kwargs):
        c_dec = args[0]
        return self._write_layer(F.concat(h_dec, c_dec, dim=1))


class SelectiveAttentionBase(HybridBlock):
    """
    Selective attention model base class.

    This class implements the attention filter construction which is used by both read and write attention modules.
    See Section 3.2 in the paper.
    """
    def __init__(self, filter_size: int, input_shape: Tuple[int, int], batch_size: int, prefix: Optional[str] = None):
        """
        :param filter_size: Size of the filter. Assumes square NxN filter.
        :param input_shape: Input shape (2-tuple: HxW).
        :param batch_size: Batch size.
        :param prefix: Prefix for block.
        """
        super().__init__(prefix=prefix)

        self._filter_size = filter_size
        self._input_shape = input_shape
        self._larger_input_dim = input_shape[0] if input_shape[0] > input_shape[1] else input_shape[1]
        self._batch_size = batch_size

        with self.name_scope():
            self._attention_params_layer = Dense(units=5)

    def _build_filter(self, F, attn_params):
        # attn params (batch x 5): gx_tilde, gy_tilde, log_var, log_delta_tilde, log_gamma (Eqn. 21)

        # Input image is M x A x B (M: batch size)
        # Filter for each image is N x N
        # F_x will be M x N x A
        # F_y will be M x N x B

        gx = ((self._input_shape[0] + 1.) / 2) * (attn_params[:, 0] + 1)
        gy = ((self._input_shape[1] + 1.) / 2) * (attn_params[:, 1] + 1)
        delta = ((self._larger_input_dim - 1) / (self._filter_size - 1)) * F.exp(attn_params[:, 3])

        # Eqn. 19 and 20.
        mu_vec = F.arange(1, self._filter_size+1) - (self._filter_size / 2.) - 0.5
        # A x N x M
        mu_x = gx + (delta * F.broadcast_to(nd.expand_dims(nd.expand_dims(mu_vec, axis=1), axis=0),
                                            (self._input_shape[0], 0, self._batch_size)))
        # B x N x M
        mu_y = gy + (delta * F.broadcast_to(nd.expand_dims(nd.expand_dims(mu_vec, axis=1), axis=0),
                                            (self._input_shape[1], 0, self._batch_size)))

        # Eqn. 25 and 26.
        # A x N x M
        Fx_grid = nd.broadcast_to(nd.expand_dims(nd.expand_dims(nd.arange(1, self._input_shape[0]+1), axis=1), axis=1),
                                  (0, self._filter_size, self._batch_size))
        Fx = F.softmax(-F.square(Fx_grid - mu_x) / (2. * F.exp(attn_params[:, 2])), axis=0)
        # B x N x M
        Fy_grid = nd.broadcast_to(nd.expand_dims(nd.expand_dims(nd.arange(1, self._input_shape[1]+1), axis=1), axis=1),
                                  (0, self._filter_size, self._batch_size))
        Fy = F.softmax(-F.square(Fy_grid - mu_y) / (2. * F.exp(attn_params[:, 2])), axis=0)

        return F.transpose(Fx), F.transpose(Fy)

    def hybrid_forward(self, F, x, *args, **kwargs):
        raise NotImplemented


class SelectiveAttentionRead(SelectiveAttentionBase):
    """
    Selective attention read block.

    Section 3.2 in the paper.
    """
    def __init__(self, filter_size: int, input_shape: Tuple[int, int], batch_size: int, prefix: Optional[str] = None):
        super().__init__(filter_size, input_shape, batch_size, prefix=prefix)

    def hybrid_forward(self, F, x, *args, **kwargs):
        err = args[0]
        h_dec = args[1]
        c_dec = args[2]
        attn_params = self._attention_params_layer(F.concat(h_dec, c_dec, dim=1))
        Fx, Fy = self._build_filter(F, attn_params)

        # Eqn. 27
        read_x = F.stack(*[F.dot(F.dot(Fx_i, x_i), Fy_i.T) for
                           Fx_i, x_i, Fy_i in zip(Fx, F.reshape(x, (-1, *self._input_shape)), Fy)], axis=0)
        read_err = F.stack(*[F.dot(F.dot(Fx_i, err_i), Fy_i.T) for
                             Fx_i, err_i, Fy_i in zip(Fx, F.reshape(err, (-1, *self._input_shape)), Fy)], axis=0)
        read = F.exp(attn_params[:, 4:5]) * F.concat(F.flatten(read_x), F.flatten(read_err))
        return read


class SelectiveAttentionWrite(SelectiveAttentionBase):
    """
    Selective attention write block.

    Section 3.2 in the paper.
    """
    def __init__(self, filter_size: int, input_shape: Tuple[int, int], batch_size: int, prefix: Optional[str] = None):
        super().__init__(filter_size, input_shape, batch_size, prefix=prefix)

        with self.name_scope():
            self._patch_layer = Dense(units=self._filter_size * self._filter_size)

    def hybrid_forward(self, F, h_dec, *args, **kwargs):
        c_dec = args[0]
        dec = F.concat(h_dec, c_dec, dim=1)
        attn_params = self._attention_params_layer(dec)
        Fx, Fy = self._build_filter(F, attn_params)

        # Eqn. 28, 29
        w = self._patch_layer(dec)
        write = F.stack(*[F.reshape(F.dot(F.dot(Fx_i.T, w_i), Fy_i), (-1,)) for
                          Fx_i, w_i, Fy_i in zip(Fx, F.reshape(w, (-1, self._filter_size, self._filter_size)), Fy)],
                        axis=0) * (1 / F.exp(attn_params[:, 4:5]))
        return write


class DRAW(HybridBlock):
    """
    DRAW model. A recurrent latent variable model.
    """
    def __init__(self, read_nn: HybridBlock, write_nn: HybridBlock, num_steps: int, batch_size: int,
                 num_rnn_units: int, input_dim: int, latent_dim: int, prefix: Optional[str] = None):
        """
        :param read_nn: Read network. This should expect input image, error image, and hidden state of decoder as
            inputs.
        :param write_nn: Write network. This should expect hidden state of decoder as input.
        :param num_steps: Number of recurrent steps (i.e., set of latent variables).
        :param batch_size: Batch size.
        :param num_rnn_units: Number of units in recurrent encoder and decoder layers.
        :param input_dim: Input dimensionality (i.e., number of elements in input).
        :param latent_dim: Dimensionality of latent space (i.e., number of elements in latent variable for each
            timestep).
        :param prefix: Prefix for block.
        """
        super().__init__(prefix=prefix)

        self._batch_size = batch_size
        self._num_steps = num_steps
        self._num_rnn_units = num_rnn_units
        self._input_dim = input_dim
        self._latent_dim = latent_dim

        with self.name_scope():
            self._read_layer = read_nn
            self._enc_rnn = LSTMCell(hidden_size=self._num_rnn_units)
            self._enc_dense = Dense(units=self._latent_dim * 2)  # output of enc rnn to latent distribution q
            self._latent_layer = NormalSamplingBlock(self._batch_size, self._latent_dim)
            self._dec_rnn = LSTMCell(hidden_size=self._num_rnn_units)
            self._write_layer = write_nn

            # learned initial values for canvas and encoder/decoder LSTM hidden states
            self._canvas_init = self.params.get('canvas', init=mx.init.Uniform(), shape=(1, self._input_dim,))

            self._enc_rnn_h_init = self.params.get('enc_rnn_h_init', init=mx.init.Uniform(),
                                                   shape=(1, self._num_rnn_units))
            self._enc_rnn_c_init = self.params.get('enc_rnn_c_init', init=mx.init.Uniform(),
                                                   shape=(1, self._num_rnn_units))

            self._dec_rnn_h_init = self.params.get('dec_rnn_h_init', init=mx.init.Uniform(),
                                                   shape=(1, self._num_rnn_units))
            self._dec_rnn_c_init = self.params.get('dec_rnn_c_init', init=mx.init.Uniform(),
                                                   shape=(1, self._num_rnn_units))

    def generate(self, n: int = 1) -> nd.NDArray:
        """
        Generate n samples from model. See Section 2.3 in paper.

        :param n: Number of samples.
        :return: n x input dim array of generated samples.
        """
        with self._canvas_init.data().context:
            canvas = nd.broadcast_to(self._canvas_init.data(), (n, 0))
            h_dec = nd.broadcast_to(self._dec_rnn_h_init.data(), (n, 0))
            c_dec = nd.broadcast_to(self._dec_rnn_c_init.data(), (n, 0))

            for i in range(self._num_steps):
                z = nd.random.normal(shape=(n, self._latent_dim))
                _, (h_dec, c_dec) = self._dec_rnn(z, [h_dec, c_dec])
                w = self._write_layer(h_dec, c_dec)
                canvas = canvas + w
        return nd.sigmoid(canvas)

    def hybrid_forward(self, F, x, *args, **kwargs):
        with x.context:
            # broadcast learned initial canvas and hidden states to batch size.
            canvas = F.broadcast_to(self._canvas_init.data(), (self._batch_size, 0))
            h_enc = F.broadcast_to(self._enc_rnn_h_init.data(), (self._batch_size, 0))
            c_enc = F.broadcast_to(self._enc_rnn_c_init.data(), (self._batch_size, 0))
            h_dec = F.broadcast_to(self._dec_rnn_h_init.data(), (self._batch_size, 0))
            c_dec = F.broadcast_to(self._dec_rnn_c_init.data(), (self._batch_size, 0))

            qs = []  # latent distributions for each step

            for i in range(self._num_steps):
                err = x - F.sigmoid(canvas)
                r = self._read_layer(x, err, h_dec, c_dec)
                _, (h_enc, c_enc) = self._enc_rnn(F.concat(r, h_dec, c_dec, dim=1), [h_enc, c_enc])

                q = self._enc_dense(h_enc)
                qs.append(q)
                z = self._latent_layer(q)

                _, (h_dec, c_dec) = self._dec_rnn(z, [h_dec, c_dec])
                w = self._write_layer(h_dec, c_dec)
                canvas = canvas + w

        # don't pass canvas through sigmoid. loss function takes care of that.
        return canvas, nd.stack(*qs, axis=0)  # qs: steps x batch x latent


class DRAWLoss(VAELoss):
    """
    Loss function for DRAW model.

    This is basically VAE loss with multiple latent variables. See Eqns. 9, 10, 11, and 12 in the paper.
    """
    def __init__(self, fit_loss: Loss, input_dim: int, latent_dim: int):
        """
        :param fit_loss: Loss used for p(x|z), i.e., reconstruction loss.
            1. Note the output of VAE is not passed through sigmoid. Make sure the loss function expects that (by
               passing from_sigmoid=False when applicable).
            2. We assume that the loss function averages over (input/output) features, rather than summing.
        :param input_dim: Input dimensionality.
        :param latent_dim: Dimensionality of latent space
        """
        super().__init__(fit_loss, input_dim, latent_dim)

    def hybrid_forward(self, F, x, *args, **kwargs):
        qs = args[0]  # steps x batch x latent
        y = args[1]  # batch x features

        fit_term = self._fit_loss(y, x) * self._input_dim  # convert avg -> sum by scaling with input dim
        kl_term = nd.sum(nd.stack(*[self._kl_loss(q) for q in qs], axis=0), axis=0)

        return fit_term + kl_term


