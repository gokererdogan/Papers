"""
An MxNet implementation of the recurrent variational autoencoder described in
    K. Gregor, I. Danihelka, A. Graves, D. J. Rezende, and D. Wierstra,
    “DRAW: A Recurrent Neural Network For Image Generation,”
    arXiv:1502.04623 [cs], Feb. 2015.

13 Aug 2018
goker erdogan
https://github.com/gokererdogan
"""
import os
from math import floor
from typing import Optional, Tuple, Union

import imageio
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
from skimage import transform
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
        return F.concat(x, err, dim=1), None


class NoAttentionWrite(HybridBlock):
    """
    Write block with no attention.

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
        return self._write_layer(F.concat(h_dec, c_dec, dim=1)), None


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

    @property
    def filter_size(self):
        return self._filter_size

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
        mu_x = gx + (delta * F.broadcast_to(F.expand_dims(F.expand_dims(mu_vec, axis=1), axis=0),
                                            (self._input_shape[0], 0, self._batch_size)))
        # B x N x M
        mu_y = gy + (delta * F.broadcast_to(F.expand_dims(F.expand_dims(mu_vec, axis=1), axis=0),
                                            (self._input_shape[1], 0, self._batch_size)))

        # Eqn. 25 and 26.
        # A x N x M
        Fx_grid = F.broadcast_to(F.expand_dims(F.expand_dims(F.arange(1, self._input_shape[0]+1), axis=1), axis=1),
                                 (0, self._filter_size, self._batch_size))
        Fx = F.softmax(-F.square(Fx_grid - mu_x) / (2. * F.exp(attn_params[:, 2])), axis=0)
        # B x N x M
        Fy_grid = F.broadcast_to(F.expand_dims(F.expand_dims(F.arange(1, self._input_shape[1]+1), axis=1), axis=1),
                                 (0, self._filter_size, self._batch_size))
        Fy = F.softmax(-F.square(Fy_grid - mu_y) / (2. * F.exp(attn_params[:, 2])), axis=0)

        return F.transpose(Fx), F.transpose(Fy)

    def hybrid_forward(self, F, x, *args, **kwargs):
        # This must return a 2-tuple of (out, attention params)
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
        return read, attn_params


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
        return write, attn_params


class DRAW(HybridBlock):
    """
    DRAW model. A recurrent latent variable model.
    """
    def __init__(self, read_nn: HybridBlock, write_nn: HybridBlock, num_steps: int, batch_size: int,
                 num_rnn_units: int, input_dim: int, latent_dim: int, ctx: mx.Context = mx.cpu(),
                 prefix: Optional[str] = None):
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
        :param ctx: Context.
        :param prefix: Prefix for block.
        """
        super().__init__(prefix=prefix)

        self._batch_size = batch_size
        self._num_steps = num_steps
        self._num_rnn_units = num_rnn_units
        self._input_dim = input_dim
        self._latent_dim = latent_dim
        self._ctx = ctx

        with self.name_scope():
            self._read_layer = read_nn
            self._enc_rnn = LSTMCell(hidden_size=self._num_rnn_units)
            self._enc_dense = Dense(units=self._latent_dim * 2)  # output of enc rnn to latent distribution q
            self._latent_layer = NormalSamplingBlock(self._batch_size, self._latent_dim)
            self._dec_rnn = LSTMCell(hidden_size=self._num_rnn_units)
            self._write_layer = write_nn

            # learned initial values for encoder/decoder LSTM hidden states
            self._enc_rnn_h_init = self.params.get('enc_rnn_h_init', init=mx.init.Uniform(),
                                                   shape=(1, self._num_rnn_units))
            self._enc_rnn_c_init = self.params.get('enc_rnn_c_init', init=mx.init.Uniform(),
                                                   shape=(1, self._num_rnn_units))

            self._dec_rnn_h_init = self.params.get('dec_rnn_h_init', init=mx.init.Uniform(),
                                                   shape=(1, self._num_rnn_units))
            self._dec_rnn_c_init = self.params.get('dec_rnn_c_init', init=mx.init.Uniform(),
                                                   shape=(1, self._num_rnn_units))

    @property
    def read_layer(self):
        return self._read_layer

    @property
    def write_layer(self):
        return self._write_layer

    def generate(self, x: nd.NDArray = None, include_intermediate: bool = False, return_attn_params: bool = False) -> \
            Union[nd.NDArray, Tuple[nd.NDArray, nd.NDArray]]:
        """
        Generate a batch of samples from model. See Section 2.3 in paper.

        If x is None, this method generates unconditional samples from the model (as explained in Section 2.3 in the
        paper).

        If x is provided, this method reconstructs the input to generate the sample. This is not really a true sample
        from the model because the model looks at the image it is trying to generate. However, this is useful for seeing
        how the model generates a particular image. (I believe this is how the figures in the paper are generated.)

        :param x: Input to generate images from. This is not really an unconditional sample from the model. This is
        :param include_intermediate: If True, samples from all timesteps (not only the last timestep) are returned.
        :param return_attn_params: If True, returns attention params along with generated samples.
        :return: n x input dim array of generated samples. If include_intermediate is True, then steps x n x input dim.
        """
        canvases = []
        attn_params = []

        canvas = nd.zeros((self._batch_size, self._input_dim), ctx=self._ctx)
        h_dec = nd.broadcast_to(self._dec_rnn_h_init.data(), (self._batch_size, 0))
        c_dec = nd.broadcast_to(self._dec_rnn_c_init.data(), (self._batch_size, 0))

        if x is not None:
            h_enc = nd.broadcast_to(self._enc_rnn_h_init.data(), (self._batch_size, 0))
            c_enc = nd.broadcast_to(self._enc_rnn_c_init.data(), (self._batch_size, 0))

        for i in range(self._num_steps):
            canvases.append(nd.sigmoid(canvas))

            if x is not None:
                err = x - nd.sigmoid(canvas)
                r, _ = self._read_layer(x, err, h_dec, c_dec)
                _, (h_enc, c_enc) = self._enc_rnn(nd.concat(r, h_dec, c_dec, dim=1), [h_enc, c_enc])

                q = self._enc_dense(h_enc)
                z = self._latent_layer(q)
            else:
                z = nd.random.normal(shape=(self._batch_size, self._latent_dim), ctx=self._ctx)

            _, (h_dec, c_dec) = self._dec_rnn(z, [h_dec, c_dec])
            w, attn_param = self._write_layer(h_dec, c_dec)
            attn_params.append(attn_param)
            canvas = canvas + w

        if include_intermediate:
            samples = nd.stack(*canvases, axis=0)
        else:
            samples = canvases[-1]

        if return_attn_params:
            return samples, nd.stack(*attn_params, axis=0)
        else:
            return samples

    def hybrid_forward(self, F, x, *args, **kwargs):
        with x.context:
            canvas = F.zeros((self._batch_size, self._input_dim), ctx=self._ctx)
            # broadcast learned hidden states to batch size.
            h_enc = F.broadcast_to(self._enc_rnn_h_init.data(), (self._batch_size, 0))
            c_enc = F.broadcast_to(self._enc_rnn_c_init.data(), (self._batch_size, 0))
            h_dec = F.broadcast_to(self._dec_rnn_h_init.data(), (self._batch_size, 0))
            c_dec = F.broadcast_to(self._dec_rnn_c_init.data(), (self._batch_size, 0))

            qs = []  # latent distributions for each step

            for i in range(self._num_steps):
                err = x - F.sigmoid(canvas)
                r, _ = self._read_layer(x, err, h_dec, c_dec)
                _, (h_enc, c_enc) = self._enc_rnn(F.concat(r, h_dec, c_dec, dim=1), [h_enc, c_enc])

                q = self._enc_dense(h_enc)
                qs.append(q)
                z = self._latent_layer(q)

                _, (h_dec, c_dec) = self._dec_rnn(z, [h_dec, c_dec])
                w, _ = self._write_layer(h_dec, c_dec)
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
            1. Note the output of DRAW is not passed through sigmoid. Make sure the loss function expects that (by
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
        kl_term = F.sum(F.stack(*[self._kl_loss(q) for q in qs], axis=0), axis=0)

        return fit_term + kl_term


def _draw_square(image: np.ndarray, center_x: int, center_y: int, width: int, thickness: int):
    """
    Draw a red square on image.

    :param image: Image to draw on.
    :param center_x: Center x (in terms of image height).
    :param center_y: Center y (in terms of image width).
    :param width: Width of square.
    :param thickness: Thickness of borders.
    :return:
    """
    # not the cleanest implementation but it does the job.
    img_h, img_w, _ = image.shape
    # top border
    row_start = floor(center_x - width/2 - thickness/2)
    row_start = min(max(row_start, 0), img_h)
    row_end = floor(center_x - width/2 + thickness/2)
    row_end = min(max(row_end, 0), img_h)

    col_start = floor(center_y - width/2 - thickness/2)
    col_start = min(max(col_start, 0), img_w)
    col_end = floor(center_y + width/2 + thickness/2)
    col_end = min(max(col_end, 0), img_w)

    image[row_start:row_end, col_start:col_end, 0] = 255

    # bottom border
    row_start = floor(center_x + width/2 - thickness/2)
    row_start = min(max(row_start, 0), img_h)
    row_end = floor(center_x + width/2 + thickness/2)
    row_end = min(max(row_end, 0), img_h)

    col_start = floor(center_y - width/2 - thickness/2)
    col_start = min(max(col_start, 0), img_w)
    col_end = floor(center_y + width/2 + thickness/2)
    col_end = min(max(col_end, 0), img_w)

    image[row_start:row_end, col_start:col_end, 0] = 255

    # left border
    row_start = floor(center_x - width/2 - thickness/2)
    row_start = min(max(row_start, 0), img_h)
    row_end = floor(center_x + width/2 + thickness/2)
    row_end = min(max(row_end, 0), img_h)

    col_start = floor(center_y - width/2 - thickness/2)
    col_start = min(max(col_start, 0), img_w)
    col_end = floor(center_y - width/2 + thickness/2)
    col_end = min(max(col_end, 0), img_w)

    image[row_start:row_end, col_start:col_end, 0] = 255

    # right border
    row_start = floor(center_x - width/2 - thickness/2)
    row_start = min(max(row_start, 0), img_h)
    row_end = floor(center_x + width/2 + thickness/2)
    row_end = min(max(row_end, 0), img_h)

    col_start = floor(center_y + width/2 - thickness/2)
    col_start = min(max(col_start, 0), img_w)
    col_end = floor(center_y + width/2 + thickness/2)
    col_end = min(max(col_end, 0), img_w)

    image[row_start:row_end, col_start:col_end, 0] = 255

    return image


def generate_sampling_gif(draw_nn: DRAW, image_shape: Union[Tuple[int, int], Tuple[int, int, int]], save_path: str,
                          save_prefix: str, from_x: nd.NDArray = None, draw_attention: bool = False,
                          scale_factor: float = 1.):
    """
    Generate animations of sampling from the given model.

    :param draw_nn: Trained DRAW model.
    :param image_shape: HxW of image.
    :param save_path: Path to save gif files in.
    :param save_prefix: Prefix for gif filenames.
    :param from_x: If provided, generate reconstructions of these images.
    :param draw_attention: If True, draws attention boxes on images. Available only if the model has selective
        attention.
    :param scale_factor: Scale images by this factor.
    :return:
    """
    if len(image_shape) == 3:
        assert image_shape[0] == 1 or image_shape[0] == 3, "Image must be grayscale or RGB."

    samples = draw_nn.generate(from_x, include_intermediate=True, return_attn_params=draw_attention)
    if draw_attention:
        samples, attn_params = samples
        attn_params = attn_params.asnumpy().transpose((1, 0, 2))
    # convert samples to images
    samples = samples.asnumpy().swapaxes(0, 1)  # batch x steps x feats
    samples = samples.reshape(samples.shape[0:2] + image_shape)  # batch x steps x h x w
    samples = (samples * 255).astype(np.uint8)

    if len(image_shape) == 3 and image_shape[0] == 1:  # has channels dimension, but there is a single channel
        # get rid of that dimension, so we can treat this as a (2D) grayscale image
        samples = samples[:, :, 0, :, :]
        image_shape = image_shape[1:]

    if len(image_shape) == 2:
        samples = np.tile(samples[:, :, :, :, None], (1, 1, 1, 1, 3))  # to rgb
    elif len(image_shape) == 3:
        # transpose CxHxW to HxWxC
        samples = samples.transpose((0, 1, 3, 4, 2))

    if draw_attention:
        larger_dim = image_shape[0] if image_shape[0] > image_shape[1] else image_shape[1]
        gx = ((image_shape[0] + 1.) / 2) * (attn_params[:, :, 0] + 1)
        gy = ((image_shape[1] + 1.) / 2) * (attn_params[:, :, 1] + 1)
        delta = ((larger_dim - 1) / (draw_nn.write_layer.filter_size - 1)) * np.exp(attn_params[:, :, 3])
        variance = np.exp(attn_params[:, :, 2])

    for i, sample in enumerate(samples):
        scaled_samples = []
        file_path = os.path.join(save_path, "{}_{}.gif".format(save_prefix, i))
        for t, sample_t in enumerate(sample):
            if draw_attention:
                if t > 0:  # skip initial canvas
                    # note it makes more sense to plot the attention window onto the image after write operation
                    # that's why we use t-1 below
                    _draw_square(image=sample_t, center_x=int(gx[i, t-1]), center_y=int(gy[i, t-1]),
                                 width=int(2 * delta[i, t-1]),
                                 thickness=int(0.06 * variance[i, t-1] + 1))  # determined heuristically
            # rescale image
            scaled_samples.append(transform.rescale(sample_t, scale=scale_factor, anti_aliasing=True, multichannel=True,
                                  preserve_range=True).astype(np.uint8))

        imageio.mimwrite(file_path, scaled_samples, duration=0.1)



