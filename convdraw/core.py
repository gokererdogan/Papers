"""
An MxNet implementation of the convolutional recurrent variational autoencoder described in
    K. Gregor, F. Besse, D. J. Rezende, I. Danihelka, and D. Wierstra,
    “Towards Conceptual Compression”
    arXiv:1604.08772 [stat.ML], Apr. 2015.

24 Sep 2018
goker erdogan
https://github.com/gokererdogan
"""
import os
from typing import Optional, Tuple, Union

import imageio
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
from mxnet.gluon.contrib.rnn import Conv2DLSTMCell
from mxnet.gluon.loss import Loss
from mxnet.gluon.nn import Conv2D, HybridBlock
from skimage import transform

from draw.core import generate_sampling_gif as draw_generate_sampling_gif
from vae.core import NormalSamplingBlock


class ConvDRAWLossKLTerm(Loss):
    """
    KL term in variational lower bound used for training ConvDraw.

    This is essentially the KL-divergence between two diagonal multivariate Gaussians.
    """
    def __init__(self, latent_dim: int):
        """
        :param latent_dim: Dimensionality of latent space.
        """
        super().__init__(weight=1, batch_axis=0)

        self._latent_dim = latent_dim

    def hybrid_forward(self, F, q, *args, **kwargs):
        p = args[0]
        # q and p are batches x latent space * 2 [mu, log_sd]
        mu_q = q[:, 0:self._latent_dim]
        mu_p = p[:, 0:self._latent_dim]
        log_sd_q = q[:, self._latent_dim:]
        log_sd_p = p[:, self._latent_dim:]

        # first term in Eq. 10. acts as a regularizer
        kl_term = 0.5 * (-self._latent_dim + F.sum(2. * log_sd_p, axis=0, exclude=True) -
                         F.sum(2. * log_sd_q, axis=0, exclude=True) +
                         F.sum(F.square((mu_q - mu_p) / F.exp(log_sd_p)), axis=0, exclude=True) +
                         F.sum(F.exp(2. * (log_sd_q - log_sd_p)), axis=0, exclude=True))

        return kl_term


class ConvDRAWLoss(Loss):
    """
    Negative variational lower bound used for training ConvDRAW.

    See Eq. 11 in the paper.
    """
    def __init__(self, fit_loss: Loss, input_dim: int, latent_shape: Tuple[int, int, int],
                 input_cost_scale: float = 1.0):
        """
        :param fit_loss: Loss used for p(x|z), i.e., reconstruction loss. Note the output of VAE is not passed through
            sigmoid. We assume that the loss function averages over (input/output) features, rather than summing.
        :param input_dim: Input dimensionality.
        :param latent_shape: Shape of latent space, CxHxW.
        """
        super().__init__(weight=1, batch_axis=0)

        assert 0. < input_cost_scale <= 1.0, "Input cost scale must be in (0, 1]"
        self._input_dim = input_dim
        self._latent_dim = int(np.prod(latent_shape))
        self._input_cost_scale = input_cost_scale

        with self.name_scope():
            self._fit_loss = fit_loss
            self._kl_loss = ConvDRAWLossKLTerm(self._latent_dim)

    def hybrid_forward(self, F, x, *args, **kwargs):
        # x: input, args[0]: latent prob. distribution, args[1]: latent prior prob. distribution,
        # args[2]: output of VAE (before sigmoid)
        qs = args[0]  # steps x batch x latent
        ps = args[1]  # steps x batch x latent
        y = args[2]  # batch x features

        fit_term = self._fit_loss(y, x) * self._input_dim  # convert avg -> sum by scaling with input dim
        kl_term = F.sum(F.stack(*[self._kl_loss(q, p) for q, p in zip(qs, ps)], axis=0), axis=0)

        return (self._input_cost_scale * fit_term) + kl_term


class ConvDRAW(HybridBlock):
    """
    ConvDRAW model. A convolutional recurrent latent variable model.

    There are a few differences between the model here and the one in the paper.
      - We use an encoder network that is applied on the difference image before it is fed into the encoder LSTM.
      - Similarly, the decoding of decoder hidden state to image is done by a decoder network, instead of a single
        convolution operation like in the paper (step 6).
      - We allow only stride=1 convolutions in LSTMs because MXNet's Convolutional LSTM implementation allows only
        stride=1 convolutions.
      - We don't learn the initial hidden states for LSTMs or the canvas. I found the results look better this way.
    """
    def __init__(self, encoder_nn: HybridBlock, decoder_nn: HybridBlock,
                 num_steps: int, batch_size: int,
                 input_shape: Tuple[int, int, int], num_latent_maps: int,
                 encoder_output_shape: Tuple[int, int, int], rnn_hidden_channels: int, kernel_size: Tuple[int, int],
                 ctx: mx.Context = mx.cpu(), prefix: Optional[str] = None):
        """
        :param encoder_nn: Encoder block that is applied to input and produces the input for the conv. LSTM encoder.
        :param decoder_nn: Decoder block that takes in the output of the conv. LSTM decoder and maps it into input
            space. This is most likely a series of transposed convolution operations.
        :param num_steps: Number of recurrent steps (i.e., set of latent variables).
        :param batch_size: Batch size.
        :param input_shape: Input shape, CxHxW.
        :param num_latent_maps: Number of latent feature maps.
        :param encoder_output_shape: Shape of the output of encoder network, CxHxW.
        :param rnn_hidden_channels: Number of channels in encoder/decoder LSTMs' hidden state.
        :param kernel_size: Kernel size for the convolution in encoder and decoder LSTMs, and latent layers.
        :param ctx: Context.
        :param prefix: Prefix for block.
        """
        super().__init__(prefix=prefix)

        self._batch_size = batch_size
        self._num_steps = num_steps
        self._input_shape = input_shape
        self._num_latent_maps = num_latent_maps
        self._encoder_output_shape = encoder_output_shape
        self._rnn_hidden_channels = rnn_hidden_channels
        self._kernel_size = kernel_size
        self._ctx = ctx

        self._rnn_hidden_shape = (self._rnn_hidden_channels, *self._encoder_output_shape[1:])
        self._latent_dim = self._num_latent_maps * int(np.prod(self._encoder_output_shape[1:]))

        with self.name_scope():
            self._enc_nn = encoder_nn

            # input to encoder rnn: x (image), difference image (eps), and hidden state of decoder (h_dec and c_dec)
            num_input_channels = (self._encoder_output_shape[0]*2 + 2*self._rnn_hidden_channels)
            pad = (self._kernel_size[0]//2, self._kernel_size[1]//2)
            self._enc_rnn = Conv2DLSTMCell(input_shape=(num_input_channels, *self._encoder_output_shape[1:]),
                                           hidden_channels=self._rnn_hidden_channels,
                                           i2h_kernel=self._kernel_size, h2h_kernel=self._kernel_size,
                                           i2h_pad=pad)  # (Step 2)
            # output of enc rnn to latent distribution q (Step 3)
            self._q_layer = Conv2D(channels=self._num_latent_maps * 2, kernel_size=self._kernel_size, padding=pad)
            self._latent_layer = NormalSamplingBlock(self._batch_size, self._latent_dim)

            self._dec_nn = decoder_nn  # (Step 6)

            # input to decoder rnn: z (latent), reconstruction (r)
            num_input_channels = (self._num_latent_maps + self._encoder_output_shape[0])
            self._dec_rnn = Conv2DLSTMCell(input_shape=(num_input_channels, *self._encoder_output_shape[1:]),
                                           hidden_channels=self._rnn_hidden_channels,
                                           i2h_kernel=self._kernel_size, h2h_kernel=self._kernel_size,
                                           i2h_pad=pad)  # (Step 5)

            # prior on q (Step 4 in the paper)
            self._p_layer = Conv2D(channels=self._num_latent_maps * 2, kernel_size=self._kernel_size, padding=pad)

    def generate(self, x: nd.NDArray = None, include_intermediate: bool = False, **kwargs) -> \
            Union[nd.NDArray, Tuple[nd.NDArray, nd.NDArray]]:
        """
        Generate a batch of samples from model. See Section 2.3 in paper.

        If x is None, this method generates unconditional samples from the model (as explained in Section 2.3 in the
        paper).

        If x is provided, this method reconstructs the input to generate the sample. This is not really a true sample
        from the model because the model looks at the image it is trying to generate. However, this is useful for seeing
        how the model generates a particular image.

        :param x: Input to generate images from. This is not really an unconditional sample from the model. This is
        :param include_intermediate: If True, samples from all timesteps (not only the last timestep) are returned.
        :return: n x *image_shape array of generated samples. If include_intermediate is True,
            then steps x n x *image_shape.
        """
        r = nd.zeros((self._batch_size, *self._input_shape), ctx=self._ctx)  # reconstruction
        h_dec = nd.zeros((self._batch_size, *self._rnn_hidden_shape), ctx=self._ctx)
        c_dec = nd.zeros((self._batch_size, *self._rnn_hidden_shape), ctx=self._ctx)

        if x is not None:
            h_enc = nd.zeros((self._batch_size, *self._rnn_hidden_shape), ctx=self._ctx)
            c_enc = nd.zeros((self._batch_size, *self._rnn_hidden_shape), ctx=self._ctx)
            encoded_x = self._enc_nn(x)

        rs = []  # sample(s) over time

        for i in range(self._num_steps):
            rs.append(nd.sigmoid(r))
            encoded_r = self._enc_nn(rs[-1])
            if x is not None:
                err = encoded_x - encoded_r
                _, (h_enc, c_enc) = self._enc_rnn(nd.concat(encoded_x, err, h_dec, c_dec, dim=1), [h_enc, c_enc])

                q = self._q_layer(h_enc)
                # convert NxCxHxW to NxF
                q = nd.reshape(q, (self._batch_size, -1))
                z = self._latent_layer(q)
            else:
                # sample from prior
                p = self._p_layer(h_dec)
                p = nd.reshape(p, (self._batch_size, -1))
                z = self._latent_layer(p)

            dec_z = nd.reshape(z, (self._batch_size, self._num_latent_maps, *self._encoder_output_shape[1:]))
            _, (h_dec, c_dec) = self._dec_rnn(nd.concat(dec_z, encoded_r, dim=1), [h_dec, c_dec])
            r = r + self._dec_nn(h_dec)

        if include_intermediate:
            samples = nd.stack(*rs, axis=0)
        else:
            samples = rs[-1]

        return samples

    def hybrid_forward(self, F, x, *args, **kwargs):
        with x.context:
            r = F.zeros((self._batch_size, *self._input_shape))  # reconstruction
            h_enc = F.zeros((self._batch_size, *self._rnn_hidden_shape))
            c_enc = F.zeros((self._batch_size, *self._rnn_hidden_shape))
            h_dec = F.zeros((self._batch_size, *self._rnn_hidden_shape))
            c_dec = F.zeros((self._batch_size, *self._rnn_hidden_shape))

            qs = []  # latent (approximate posterior) distributions for each step
            ps = []  # prior distributions of latents for each step

            encoded_x = self._enc_nn(x)
            for i in range(self._num_steps):
                encoded_r = self._enc_nn(F.sigmoid(r))
                err = encoded_x - encoded_r

                _, (h_enc, c_enc) = self._enc_rnn(F.concat(encoded_x, err, h_dec, c_dec, dim=1), [h_enc, c_enc])

                q = self._q_layer(h_enc)
                # convert NxCxHxW to NxF
                q = F.reshape(q, (self._batch_size, -1))
                qs.append(q)
                z = self._latent_layer(q)

                p = self._p_layer(h_dec)
                p = F.reshape(p, (self._batch_size, -1))
                ps.append(p)

                dec_z = F.reshape(z, (self._batch_size, self._num_latent_maps, *self._encoder_output_shape[1:]))
                _, (h_dec, c_dec) = self._dec_rnn(F.concat(dec_z, encoded_r, dim=1), [h_dec, c_dec])

                r = r + self._dec_nn(h_dec)

        # don't pass reconstruction through sigmoid. loss function takes care of that.
        return r, nd.stack(*qs, axis=0), nd.stack(*ps, axis=0)  # qs and ps: steps x batch x latent


def generate_sampling_gif(conv_draw_nn: ConvDRAW, image_shape: Tuple[int, int, int], save_path: str, save_prefix: str,
                          from_x: nd.NDArray = None, scale_factor: float = 1.):
    """
    Generate animations of sampling from the given model and save them to disk.

    :param conv_draw_nn: Trained ConvDRAW model.
    :param image_shape: HxW of image.
    :param save_path: Path to save gif files in.
    :param save_prefix: Prefix for gif filenames.
    :param from_x: If provided, generate reconstructions of these images.
    :param scale_factor: Scale images by this factor.
    :return:
    """
    return draw_generate_sampling_gif(conv_draw_nn, image_shape, save_path, save_prefix,
                                      from_x=from_x, draw_attention=False, scale_factor=scale_factor)


def generate_samples(conv_draw_nn: ConvDRAW, image_shape: Tuple[int, int, int], save_path: str, save_prefix: str,
                     scale_factor: float = 1.):
    """
    Generate samples and save them to disk.

    :param conv_draw_nn: Trained ConvDRAW model.
    :param image_shape: HxW of image.
    :param save_path: Path to save gif files in.
    :param save_prefix: Prefix for gif filenames.
    :param scale_factor: Scale images by this factor.
    :return:
    """
    samples = conv_draw_nn.generate(None, include_intermediate=False)

    # convert samples to binary images (flip black and white)
    # we use 0.2 for black-white threshold (instead of 0.5). this produces visually more appealing results.
    samples = ((samples.asnumpy() < 0.5) * 255).astype(np.uint8)
    samples = samples.transpose((0, 2, 3, 1))  # from CxHxW to HxWxC
    if image_shape[0] == 1:
        samples = np.tile(samples, (1, 1, 1, 3))  # to rgb

    for i, sample in enumerate(samples):
        file_path = os.path.join(save_path, "{}_{}.png".format(save_prefix, i))
        # rescale image
        scaled_sample = transform.rescale(sample, scale=scale_factor, anti_aliasing=True, multichannel=True,
                                          preserve_range=True).astype(np.uint8)

        imageio.imwrite(file_path, scaled_sample)
