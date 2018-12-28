"""
An MxNet implementation of the convolutional recurrent variational autoencoder described in
    S. M. Ali Eslami et al.
    “Neural scene representation and rendering”
    2018.

23 Dec 2018
goker erdogan
https://github.com/gokererdogan
"""
from typing import Optional, Tuple, Union

import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
from mxnet.gluon.contrib.rnn import Conv2DLSTMCell
from mxnet.gluon.nn import Conv2D, HybridBlock

from common import WithELU
from convdraw.core import ConvDRAWLoss
from vae.core import NormalSamplingBlock

_NUM_CAMERA_PARAMS = 7


GQNLoss = ConvDRAWLoss


class NormalFitLoss(HybridBlock):
    """
    Loss function for output with Gaussian distribution.

    This basically the (mean) negative log prob of a diagonal Normal distribution.
    """
    def __init__(self, std: float, prefix: str = None):
        super().__init__(prefix=prefix)

        self._std = std

    def hybrid_forward(self, F, x, *args, **kwargs):
        # inputs: prediction, true
        pred = x
        true = args[0]

        t1 = -0.5 * np.log(2*np.pi)
        t2 = -np.log(self._std)
        t3 = -0.5 * F.mean(F.square((pred - true) / self._std), axis=0, exclude=True)
        prob = t1 + t2 + t3
        return -prob


class RepresentationNetworkTower(HybridBlock):
    """
    Representation network referred to as the tower architecture in the paper.

    This network uses ELUs instead of ReLUs as used in the paper.
    """
    def __init__(self, batch_size: int, num_camera_params: int = None, prefix: Optional[str] = None):
        super().__init__(prefix=prefix)

        self._batch_size = batch_size
        self._img_shape = (3, 64, 64)
        self._num_camera_params = num_camera_params or _NUM_CAMERA_PARAMS

        with self.name_scope():
            # note the number of filters in the layers before residual connections are different from the ones in the
            # paper.
            # it's not clear how the residual connections can work with the number of filters given in the paper
            # the number of feature maps in the input and output of the conv layers inside the residual connections
            # don't match. this is either a typo in the paper or there are additional convolution operations inside the
            # residual connections to match the sizes.
            # below, we'll just change the number of filters to make the residual connections work.
            self._conv1 = WithELU(Conv2D(channels=128, kernel_size=(2, 2), strides=(2, 2)))
            self._conv2 = WithELU(Conv2D(channels=128, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)))
            self._conv3 = WithELU(Conv2D(channels=128-self._num_camera_params, kernel_size=(2, 2), strides=(2, 2)))
            self._conv4 = WithELU(Conv2D(channels=128, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)))
            self._conv5 = WithELU(Conv2D(channels=256, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)))
            self._conv6 = WithELU(Conv2D(channels=256, kernel_size=(1, 1), strides=(1, 1)))

    def get_output_shape(self):
        return self._batch_size, 256, 16, 16

    def hybrid_forward(self, F, x, *args, **kwargs):
        # inputs: x: frames (BxKxCxHxW), args[0]: cameras (viewpoints) (BxKxF)
        # K is the context dimension
        v = args[0]

        # get rid of context dimension so we can feed the images to conv layers
        x = F.reshape(x, (-1, *self._img_shape))
        v = F.reshape(v, (-1, self._num_camera_params))

        y1 = self._conv1(x)  # k: 2x2, s: 2x2
        y2 = self._conv2(y1)  # k: 3x3, s: 1x1
        y2 = y2 + y1  # residual
        y3 = self._conv3(y2)  # k: 2x2, s: 2x2

        # concat camera information
        v = F.broadcast_to(F.expand_dims(F.expand_dims(v, axis=-1), axis=-1), (0, 0, 16, 16))
        y3 = F.concat(y3, v, dim=1)

        y4 = self._conv4(y3)  # k: 3x3, s: 1x1
        y4 = y4 + y3  # residual
        y5 = self._conv5(y4)  # k: 3x3, s: 1x1
        y6 = self._conv6(y5)  # k: 1x1, s: 1x1

        # sum over context
        r = F.sum(F.reshape(y6, (self._batch_size, -1, 256, 16, 16)), axis=1)
        return r


class GenerativeQueryNetwork(HybridBlock):
    """
    Generative Query Network (GQN) model. A convolutional recurrent latent variable model.

    This model is the same with ConvDRAw model except GQN is a conditional generative model and hence it expects an
    additional neural network (called representation network) to process the conditioning variables.

    NOTE
        - We need an encoder network to map the query image to the same size as the output of representation network.
          Otherwise, we wouldn't be able to concatenate these two and feed them into the inference LSTM. (Eq. S20).
          This encoder is never mentioned explicitly in the paper.
        - Similarly, a decoder network maps the output of generation LSTM to an image with the right size. (Eq. S16).
        - GQN doesn't calculate an error image in contrast to the ConvDRAW and DRAW models. It uses the image directly.
        - GQN generation LSTM (i.e., decoder LSTM) doesn't take the canvas as input in contrast to ConvDRAW model.
    TODO: add docstring
    """
    def __init__(self, representation_nn: HybridBlock, encoder_nn: HybridBlock, decoder_nn: HybridBlock,
                 num_steps: int, batch_size: int,
                 input_shape: Tuple[int, int, int], num_latent_maps: int,
                 representation_shape: Tuple[int, int, int], encoder_output_shape: Tuple[int, int, int],
                 rnn_hidden_channels: int, kernel_size: Tuple[int, int], num_camera_params: int = None,
                 ctx: mx.Context = mx.cpu(), prefix: Optional[str] = None):
        """
        :param representation_nn: Representation network. The context images are processed with this network to produce
            output r in the paper.
        :param encoder_nn: Encoder block that is applied to input and produces the input for the conv. LSTM encoder.
        :param decoder_nn: Decoder block that takes in the output of the conv. LSTM decoder and maps it into input
            space. This is most likely a series of transposed convolution operations.
        :param num_steps: Number of recurrent steps (i.e., set of latent variables).
        :param batch_size: Batch size.
        :param input_shape: Input shape, CxHxW.
        :param num_latent_maps: Number of latent feature maps.
        :param representation_shape: shape of the output of representation network, CxHxW.
        :param encoder_output_shape: Shape of the output of encoder network, CxHxW.
            NOTE that the representation network output and encoder output MUST have the same H and W (because these
            are concatenated to feed into the conv LSTMs).
        :param rnn_hidden_channels: Number of channels in encoder/decoder LSTMs' hidden state.
        :param kernel_size: Kernel size for the convolution in encoder and decoder LSTMs, and latent layers.
        :param ctx: Context.
        :param prefix: Prefix for block.
        """
        super().__init__(prefix=prefix)

        assert representation_shape[1:] == encoder_output_shape[1:], "Encoder and representation network must output " \
                                                                     "images of the same H and W."

        self._batch_size = batch_size
        self._num_steps = num_steps
        self._input_shape = input_shape
        self._num_latent_maps = num_latent_maps
        self._encoder_output_shape = encoder_output_shape
        self._representation_shape = representation_shape
        self._rnn_hidden_channels = rnn_hidden_channels
        self._kernel_size = kernel_size
        self._ctx = ctx

        self._num_camera_params = num_camera_params or _NUM_CAMERA_PARAMS

        self._rnn_hidden_shape = (self._rnn_hidden_channels, *self._encoder_output_shape[1:])
        self._latent_dim = self._num_latent_maps * int(np.prod(self._encoder_output_shape[1:]))

        with self.name_scope():
            self._representation_nn = representation_nn  # r
            self._enc_nn = encoder_nn

            # input to inference LSTM: x^q (image), v^q (camera), r, u (canvas), h^g (hidden state of generation LSTM)
            num_input_channels = (2*self._encoder_output_shape[0] + self._num_camera_params +
                                  self._representation_shape[0] + self._rnn_hidden_channels)
            pad = (self._kernel_size[0]//2, self._kernel_size[1]//2)
            self._inf_rnn = Conv2DLSTMCell(input_shape=(num_input_channels, *self._encoder_output_shape[1:]),
                                           hidden_channels=self._rnn_hidden_channels,
                                           i2h_kernel=self._kernel_size, h2h_kernel=self._kernel_size,
                                           i2h_pad=pad)  # C^e
            # output of inference LSTM to latent distribution q (Eq. S21)
            self._q_layer = Conv2D(channels=self._num_latent_maps * 2, kernel_size=self._kernel_size, padding=pad)
            self._latent_layer = NormalSamplingBlock(self._batch_size, self._latent_dim)

            self._dec_nn = decoder_nn  # Delta network in Eq. S16.

            # input to generation LSTM: v^q (camera), r, z (latent)
            num_input_channels = (self._num_camera_params + self._representation_shape[0] + self._num_latent_maps)
            self._gen_rnn = Conv2DLSTMCell(input_shape=(num_input_channels, *self._encoder_output_shape[1:]),
                                           hidden_channels=self._rnn_hidden_channels,
                                           i2h_kernel=self._kernel_size, h2h_kernel=self._kernel_size,
                                           i2h_pad=pad)  # C^g

            # prior on q (Eq. S11 in the paper)
            self._p_layer = Conv2D(channels=self._num_latent_maps * 2, kernel_size=self._kernel_size, padding=pad)

    def generate(self, v_q: nd.NDArray, x_context: nd.NDArray, v_context: nd.NDArray,
                 include_intermediate: bool = False, **kwargs) -> Union[nd.NDArray, Tuple[nd.NDArray, nd.NDArray]]:
        """
        Generate a batch of samples from model. See Algorithm S3 in paper.

        :param v_q: Query view camera info.
        :param x_context: Context frames.
        :param v_context: Context camera info.
        :param include_intermediate: If True, samples from all timesteps (not only the last timestep) are returned.
        :return: n x *image_shape array of generated samples. If include_intermediate is True,
            then steps x n x *image_shape.
        """
        u = nd.zeros((self._batch_size, *self._input_shape), ctx=self._ctx)  # canvas (reconstruction)
        h_dec = nd.zeros((self._batch_size, *self._rnn_hidden_shape), ctx=self._ctx)
        c_dec = nd.zeros((self._batch_size, *self._rnn_hidden_shape), ctx=self._ctx)

        # reshape camera information so we can concat it to image data
        v_q = nd.broadcast_to(nd.expand_dims(nd.expand_dims(v_q, axis=-1), axis=-1),
                              (0, 0, *self._encoder_output_shape[1:]))

        us = []  # sample(s) over time

        r = self._representation_nn(x_context, v_context)
        for i in range(self._num_steps):
            us.append(nd.sigmoid(u))

            # Eq. S11
            p = self._p_layer(h_dec)
            p = nd.reshape(p, (self._batch_size, -1))
            z = self._latent_layer(p)

            gen_z = nd.reshape(z, (self._batch_size, self._num_latent_maps, *self._encoder_output_shape[1:]))
            _, (h_dec, c_dec) = self._gen_rnn(nd.concat(gen_z, v_q, r, dim=1), [h_dec, c_dec])

            u = u + self._dec_nn(h_dec)

        if include_intermediate:
            samples = nd.stack(*us, axis=0)
        else:
            samples = us[-1]

        return nd.clip(samples, a_min=0.0, a_max=1.0)

    def hybrid_forward(self, F, x, *args, **kwargs):
        # inputs: query frame x^q, query camera v^q, context frames x^{1..M}, context cameras v^{1..M}
        x_q = x
        v_q = args[0]
        x_context = args[1]
        v_context = args[2]

        # reshape camera information so we can concat it to image data
        v_q = F.broadcast_to(F.expand_dims(F.expand_dims(v_q, axis=-1), axis=-1),
                             (0, 0, *self._encoder_output_shape[1:]))

        with x_q.context:
            u = F.zeros((self._batch_size, *self._input_shape))  # canvas (reconstruction)
            h_enc = F.zeros((self._batch_size, *self._rnn_hidden_shape))
            c_enc = F.zeros((self._batch_size, *self._rnn_hidden_shape))
            h_dec = F.zeros((self._batch_size, *self._rnn_hidden_shape))
            c_dec = F.zeros((self._batch_size, *self._rnn_hidden_shape))

            qs = []  # latent (approximate posterior) distributions for each step
            ps = []  # prior distributions of latents for each step

            r = self._representation_nn(x_context, v_context)
            encoded_x_q = self._enc_nn(x_q)
            for i in range(self._num_steps):
                encoded_u = self._enc_nn(F.sigmoid(u))

                # Eq. S20
                _, (h_enc, c_enc) = self._inf_rnn(F.concat(encoded_x_q, v_q, encoded_u, r, h_dec, dim=1),
                                                  [h_enc, c_enc])

                # Eq. S21
                q = self._q_layer(h_enc)

                # convert NxCxHxW to NxF
                q = F.reshape(q, (self._batch_size, -1))
                qs.append(q)
                z = self._latent_layer(q)

                # Eq. S11
                p = self._p_layer(h_dec)
                p = F.reshape(p, (self._batch_size, -1))
                ps.append(p)

                gen_z = F.reshape(z, (self._batch_size, self._num_latent_maps, *self._encoder_output_shape[1:]))
                _, (h_dec, c_dec) = self._gen_rnn(F.concat(gen_z, v_q, r, dim=1), [h_dec, c_dec])

                u = u + self._dec_nn(h_dec)

        return F.sigmoid(u), nd.stack(*qs, axis=0), nd.stack(*ps, axis=0)  # qs and ps: steps x batch x latent
