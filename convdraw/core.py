"""
An MxNet implementation of the convolutional recurrent variational autoencoder described in
    K. Gregor, F. Besse, D. J. Rezende, I. Danihelka, and D. Wierstra,
    “Towards Conceptual Compression”
    arXiv:1604.08772 [stat.ML], Apr. 2015.

24 Sep 2018
goker erdogan
https://github.com/gokererdogan
"""
from typing import Optional, Tuple

import mxnet as mx
import mxnet.ndarray as nd
from mxnet.gluon.contrib.rnn import Conv2DLSTMCell
from mxnet.gluon.nn import Dense, HybridBlock

from draw.core import generate_sampling_gif as draw_generate_sampling_gif
from vae.core import NormalSamplingBlock


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
                 input_shape: Tuple[int, int, int], latent_dim: int,
                 encoder_output_shape: Tuple[int, int, int], rnn_hidden_channels: int, kernel_size: Tuple[int, int],
                 ctx: mx.Context = mx.cpu(), prefix: Optional[str] = None):
        """
        :param encoder_nn: Encoder block that is applied to input and produces the input for the conv. LSTM encoder.
        :param decoder_nn: Decoder block that takes in the output of the conv. LSTM decoder and maps it into input
            space. This is most likely a series of transposed convolution operations.
        :param num_steps: Number of recurrent steps (i.e., set of latent variables).
        :param batch_size: Batch size.
        :param input_shape: Input shape, CxHxW.
        :param latent_dim: Number of latent features.
        :param encoder_output_shape: Shape of the output of encoder network, CxHxW.
        :param rnn_hidden_channels: Number of channels in encoder/decoder LSTMs' hidden state.
        :param kernel_size: Kernel size for the convolution in encoder and decoder LSTMs.
        :param ctx: Context.
        :param prefix: Prefix for block.
        """
        super().__init__(prefix=prefix)

        self._batch_size = batch_size
        self._num_steps = num_steps
        self._input_shape = input_shape
        self._latent_dim = latent_dim
        self._encoder_output_shape = encoder_output_shape
        self._rnn_hidden_channels = rnn_hidden_channels
        self._kernel_size = kernel_size
        self._ctx = ctx

        self._rnn_hidden_shape = (self._rnn_hidden_channels, *self._encoder_output_shape[1:])

        with self.name_scope():
            self._enc_nn = encoder_nn

            # input to encoder rnn: x (image), difference image (eps), and hidden state of decoder (h_dec and c_dec)
            num_input_channels = (self._encoder_output_shape[0]*2 + 2*self._rnn_hidden_channels)
            self._enc_rnn = Conv2DLSTMCell(input_shape=(num_input_channels, *self._encoder_output_shape[1:]),
                                           hidden_channels=self._rnn_hidden_channels,
                                           i2h_kernel=self._kernel_size, h2h_kernel=self._kernel_size,
                                           i2h_pad=(self._kernel_size[0]//2, self._kernel_size[1]//2))  # (Step 2)
            self._enc_dense = Dense(units=self._latent_dim * 2)  # output of enc rnn to latent distribution q (Step 3)
            self._latent_layer = NormalSamplingBlock(self._batch_size, self._latent_dim)

            self._dec_nn = decoder_nn  # (Step 6)

            # input to decoder rnn: z (latent), reconstruction (r)
            num_input_channels = (self._latent_dim + self._encoder_output_shape[0])
            self._dec_rnn = Conv2DLSTMCell(input_shape=(num_input_channels, *self._encoder_output_shape[1:]),
                                           hidden_channels=self._rnn_hidden_channels,
                                           i2h_kernel=self._kernel_size, h2h_kernel=self._kernel_size,
                                           i2h_pad=(self._kernel_size[0]//2, self._kernel_size[1]//2))  # (Step 5)
            self._dec_dense = Dense(units=self._latent_dim * 2)  # prior on q (Step 4 in the paper)

    # def generate(self, x: nd.NDArray = None, include_intermediate: bool = False, return_attn_params: bool = False) -> \
    #         Union[nd.NDArray, Tuple[nd.NDArray, nd.NDArray]]:
    #     """
    #     Generate a batch of samples from model. See Section 2.3 in paper.
    #
    #     If x is None, this method generates unconditional samples from the model (as explained in Section 2.3 in the
    #     paper).
    #
    #     If x is provided, this method reconstructs the input to generate the sample. This is not really a true sample
    #     from the model because the model looks at the image it is trying to generate. However, this is useful for seeing
    #     how the model generates a particular image. (I believe this is how the figures in the paper are generated.)
    #
    #     :param x: Input to generate images from. This is not really an unconditional sample from the model. This is
    #     :param include_intermediate: If True, samples from all timesteps (not only the last timestep) are returned.
    #     :param return_attn_params: If True, returns attention params along with generated samples.
    #     :return: n x input dim array of generated samples. If include_intermediate is True, then steps x n x input dim.
    #     """
    #     canvases = []
    #     attn_params = []
    #
    #     canvas = nd.zeros((self._batch_size, *self._input_shape), ctx=self._ctx)
    #     h_dec = nd.zeros((self._batch_size, *self._input_shape), ctx=self._ctx)
    #     c_dec = nd.zeros((self._batch_size, *self._input_shape), ctx=self._ctx)
    #
    #     if x is not None:
    #         h_enc = nd.broadcast_to(self._enc_rnn_h_init.data(), (self._batch_size, 0))
    #         c_enc = nd.broadcast_to(self._enc_rnn_c_init.data(), (self._batch_size, 0))
    #
    #     for i in range(self._num_steps):
    #         canvases.append(nd.sigmoid(canvas))
    #
    #         if x is not None:
    #             err = x - nd.sigmoid(canvas)
    #             r, _ = self._read_layer(x, err, h_dec, c_dec)
    #             _, (h_enc, c_enc) = self._enc_rnn(nd.concat(r, h_dec, c_dec, dim=1), [h_enc, c_enc])
    #
    #             q = self._enc_dense(h_enc)
    #             z = self._latent_layer(q)
    #         else:
    #             z = nd.random.normal(shape=(self._batch_size, self._latent_dim), ctx=self._ctx)
    #
    #         _, (h_dec, c_dec) = self._dec_rnn(z, [h_dec, c_dec])
    #         w, attn_param = self._write_layer(h_dec, c_dec)
    #         attn_params.append(attn_param)
    #         canvas = canvas + w
    #
    #     if include_intermediate:
    #         samples = nd.stack(*canvases, axis=0)
    #     else:
    #         samples = canvases[-1]
    #
    #     if return_attn_params:
    #         return samples, nd.stack(*attn_params, axis=0)
    #     else:
    #         return samples

    def hybrid_forward(self, F, x, *args, **kwargs):
        with x.context:
            r = F.zeros((self._batch_size, *self._input_shape), ctx=self._ctx)  # reconstruction
            h_enc = F.zeros((self._batch_size, *self._rnn_hidden_shape))
            c_enc = F.zeros((self._batch_size, *self._rnn_hidden_shape))
            h_dec = F.zeros((self._batch_size, *self._rnn_hidden_shape))
            c_dec = F.zeros((self._batch_size, *self._rnn_hidden_shape))

            qs = []  # latent (approximate posterior) distributions for each step
            ps = []  # prior distributions of latents for each step

            for i in range(self._num_steps):
                encoded_x = self._enc_nn(x)
                encoded_r = self._enc_nn(r)
                err = encoded_x - encoded_r

                _, (h_enc, c_enc) = self._enc_rnn(F.concat(encoded_x, err, h_dec, c_dec, dim=1), [h_enc, c_enc])

                q = self._enc_dense(h_enc)
                qs.append(q)
                z = self._latent_layer(q)

                p = self._dec_dense(h_dec)
                ps.append(p)

                dec_z = F.broadcast_to(F.expand_dims(F.expand_dims(z, axis=-1), axis=-1),
                                       (self._batch_size, self._latent_dim, *self._rnn_hidden_shape[1:]))
                _, (h_dec, c_dec) = self._dec_rnn(F.concat(dec_z, encoded_r, dim=1), [h_dec, c_dec])

                r = r + self._dec_nn(h_dec)

        # don't pass reconstruction through sigmoid. loss function takes care of that.
        return r, nd.stack(*qs, axis=0), nd.stack(*ps, axis=0)  # qs and ps: steps x batch x latent

#
# class ConvDRAWLoss(VAELoss):
#     """
#     TODO: implement this
#     Loss function for ConvDRAW model.
#
#     This is basically VAE loss with multiple latent variables. See Eqns. 9, 10, 11, and 12 in the paper.
#     """
#     def __init__(self, fit_loss: Loss, input_dim: int, latent_dim: int):
#         """
#         :param fit_loss: Loss used for p(x|z), i.e., reconstruction loss.
#             1. Note the output of VAE is not passed through sigmoid. Make sure the loss function expects that (by
#                passing from_sigmoid=False when applicable).
#             2. We assume that the loss function averages over (input/output) features, rather than summing.
#         :param input_dim: Input dimensionality.
#         :param latent_dim: Dimensionality of latent space
#         """
#         super().__init__(fit_loss, input_dim, latent_dim)
#
#     def hybrid_forward(self, F, x, *args, **kwargs):
#         qs = args[0]  # steps x batch x latent
#         y = args[1]  # batch x features
#
#         fit_term = self._fit_loss(y, x) * self._input_dim  # convert avg -> sum by scaling with input dim
#         kl_term = F.sum(F.stack(*[self._kl_loss(q) for q in qs], axis=0), axis=0)
#
#         return fit_term + kl_term


def generate_sampling_gif(conv_draw_nn: ConvDRAW, image_shape: Tuple[int, int], save_path: str, save_prefix: str,
                          from_x: nd.NDArray = None, scale_factor: float = 1.):
    """
    Generate animations of sampling from the given model.

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
