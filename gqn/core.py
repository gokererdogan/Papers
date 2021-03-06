"""
An MxNet implementation of the convolutional recurrent variational autoencoder described in
    S. M. Ali Eslami et al.
    “Neural scene representation and rendering”
    2018.

23 Dec 2018
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
from mxnet.gluon.nn import Conv2D, HybridBlock
from skimage import transform

from common import WithELU, HyperparamScheduler, GQNDataIter
from convdraw.core import ConvDRAWLoss
from vae.core import NormalSamplingBlock

_NUM_CAMERA_PARAMS = 7


GQNLoss = ConvDRAWLoss


class PixelStdScheduler(HyperparamScheduler):
    """
    Scheduler for pixel standard deviation.

    See Table S1 in the supplementary materials.
    """
    def __init__(self, std_initial: float, std_final: float, num_train_samples: int):
        self._std_initial = std_initial
        self._std_final = std_final
        self._num_train_samples = num_train_samples

    def get_hyperparams(self, samples_processed: int):
        remain_ratio = 1 - (samples_processed / float(self._num_train_samples))
        std = max(self._std_final, (self._std_final + (self._std_initial - self._std_final)*remain_ratio))
        return {'pixel_std': std}


class NormalFitLoss(HybridBlock):
    """
    Loss function for output with Gaussian distribution.

    This is basically the (mean) negative log prob of a diagonal Normal distribution.
    """
    def __init__(self, std: float, prefix: str = None):
        super().__init__(prefix=prefix)

        self._std = std

    def set_std(self, std: float):
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
    """
    def __init__(self, representation_nn: HybridBlock, downsample_nn: HybridBlock, upsample_nn: HybridBlock,
                 num_steps: int, batch_size: int,
                 input_shape: Tuple[int, int, int], num_latent_maps: int,
                 representation_shape: Tuple[int, int, int],
                 downsample_output_shape: Tuple[int, int, int],
                 upsample_output_shape: Tuple[int, int, int],
                 rnn_hidden_channels: int, kernel_size: Tuple[int, int], num_camera_params: int = None,
                 ctx: mx.Context = mx.cpu(), prefix: Optional[str] = None):
        """
        :param representation_nn: Representation network. The context images are processed with this network to produce
            output r in the paper.
        :param downsample_nn: Downsampling block that is applied to input and produces the input for the conv. LSTM
            encoder.
        :param upsample_nn: Upsampling block that takes in the output of the conv. LSTM decoder and maps it into input
            space. This is most likely a series of transposed convolution operations.
        :param num_steps: Number of recurrent steps (i.e., set of latent variables).
        :param batch_size: Batch size.
        :param input_shape: Input shape, CxHxW.
        :param num_latent_maps: Number of latent feature maps.
        :param representation_shape: shape of the output of representation network, CxHxW.
        :param downsample_output_shape: Shape of the output of upsampling network, CxHxW.
            NOTE that the representation network output and upsampling network output MUST have the same H and W
            (because these are concatenated to feed into the conv LSTMs).
        :param upsample_output_shape: Output shape of upsampling network. H and W MUST be the same with input H and W.
        :param rnn_hidden_channels: Number of channels in encoder/decoder LSTMs' hidden state.
        :param kernel_size: Kernel size for the convolution in encoder and decoder LSTMs, and latent layers.
        :param ctx: Context.
        :param prefix: Prefix for block.
        """
        super().__init__(prefix=prefix)

        assert representation_shape[1:] == downsample_output_shape[1:], "Downsample and representation networks must " \
                                                                        "output images of the same H and W."
        assert input_shape[1:] == upsample_output_shape[1:], "Upsample network output must have the same H and W " \
                                                             "with input."

        self._batch_size = batch_size
        self._num_steps = num_steps
        self._input_shape = input_shape
        self._num_latent_maps = num_latent_maps
        self._downsample_output_shape = downsample_output_shape
        self._upsample_output_shape = upsample_output_shape
        self._representation_shape = representation_shape
        self._rnn_hidden_channels = rnn_hidden_channels
        self._kernel_size = kernel_size
        self._ctx = ctx

        self._num_camera_params = num_camera_params or _NUM_CAMERA_PARAMS

        self._rnn_hidden_shape = (self._rnn_hidden_channels, *self._downsample_output_shape[1:])
        self._latent_dim = self._num_latent_maps * int(np.prod(self._downsample_output_shape[1:]))

        with self.name_scope():
            self._representation_nn = representation_nn  # r
            self._downsample_nn = downsample_nn

            # input to inference LSTM: x^q (image), u (canvas), v^q (camera), r, h^g (hidden state of generation LSTM)
            num_input_channels = (self._downsample_output_shape[0] + self._num_camera_params +
                                  self._representation_shape[0] + self._rnn_hidden_channels)
            pad = (self._kernel_size[0]//2, self._kernel_size[1]//2)
            self._inf_rnn = Conv2DLSTMCell(input_shape=(num_input_channels, *self._downsample_output_shape[1:]),
                                           hidden_channels=self._rnn_hidden_channels,
                                           i2h_kernel=self._kernel_size, h2h_kernel=self._kernel_size,
                                           i2h_pad=pad)  # C^e
            # output of inference LSTM to latent distribution q (Eq. S21)
            self._q_layer = Conv2D(channels=self._num_latent_maps * 2, kernel_size=(1, 1), padding=(0, 0))
            self._latent_layer = NormalSamplingBlock(self._batch_size, self._latent_dim)

            self._upsample_nn = upsample_nn  # Delta network in Eq. S16.

            # input to generation LSTM: v^q (camera), r, z (latent)
            num_input_channels = (self._num_camera_params + self._representation_shape[0] + self._num_latent_maps)
            self._gen_rnn = Conv2DLSTMCell(input_shape=(num_input_channels, *self._downsample_output_shape[1:]),
                                           hidden_channels=self._rnn_hidden_channels,
                                           i2h_kernel=self._kernel_size, h2h_kernel=self._kernel_size,
                                           i2h_pad=pad)  # C^g

            # prior on q (Eq. S11 in the paper)
            self._p_layer = Conv2D(channels=self._num_latent_maps * 2, kernel_size=(1, 1), padding=(0, 0))

            # eta^g in Eq. S14. Maps u to output.
            self._out_layer = Conv2D(channels=self._input_shape[0], kernel_size=(1, 1), padding=(0, 0))

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
        u = nd.zeros((self._batch_size, *self._upsample_output_shape), ctx=self._ctx)  # canvas (reconstruction)
        h_dec = nd.zeros((self._batch_size, *self._rnn_hidden_shape), ctx=self._ctx)
        c_dec = nd.zeros((self._batch_size, *self._rnn_hidden_shape), ctx=self._ctx)

        # reshape camera information so we can concat it to image data
        v_q = nd.broadcast_to(nd.expand_dims(nd.expand_dims(v_q, axis=-1), axis=-1),
                              (0, 0, *self._downsample_output_shape[1:]))

        outs = []  # sample(s) over time

        r = self._representation_nn(x_context, v_context)
        for i in range(self._num_steps):
            outs.append(self._out_layer(u))

            # Eq. S11
            p = self._p_layer(h_dec)
            p = nd.reshape(p, (self._batch_size, -1))
            z = self._latent_layer(p)

            gen_z = nd.reshape(z, (self._batch_size, self._num_latent_maps, *self._downsample_output_shape[1:]))
            _, (h_dec, c_dec) = self._gen_rnn(nd.concat(gen_z, v_q, r, dim=1), [h_dec, c_dec])

            u = u + self._upsample_nn(h_dec)

        outs.append(self._out_layer(u))

        if include_intermediate:
            samples = nd.stack(*outs, axis=0)
        else:
            samples = outs[-1]

        return nd.clip(samples, a_min=0.0, a_max=1.0)

    def hybrid_forward(self, F, x, *args, **kwargs):
        # inputs: query frame x^q, query camera v^q, context frames x^{1..M}, context cameras v^{1..M}
        x_q = x
        v_q = args[0]
        x_context = args[1]
        v_context = args[2]

        # reshape camera information so we can concat it to image data
        v_q = F.broadcast_to(F.expand_dims(F.expand_dims(v_q, axis=-1), axis=-1),
                             (0, 0, *self._downsample_output_shape[1:]))

        with x_q.context:
            u = F.zeros((self._batch_size, *self._upsample_output_shape))  # canvas (reconstruction)
            h_enc = F.zeros((self._batch_size, *self._rnn_hidden_shape))
            c_enc = F.zeros((self._batch_size, *self._rnn_hidden_shape))
            h_dec = F.zeros((self._batch_size, *self._rnn_hidden_shape))
            c_dec = F.zeros((self._batch_size, *self._rnn_hidden_shape))

            qs = []  # latent (approximate posterior) distributions for each step
            ps = []  # prior distributions of latents for each step

            r = self._representation_nn(x_context, v_context)
            for i in range(self._num_steps):
                encoded_x_q_and_u = self._downsample_nn(F.concat(x_q, u, dim=1))

                # Eq. S20
                _, (h_enc, c_enc) = self._inf_rnn(F.concat(encoded_x_q_and_u, v_q, r, h_dec, dim=1),
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

                gen_z = F.reshape(z, (self._batch_size, self._num_latent_maps, *self._downsample_output_shape[1:]))
                _, (h_dec, c_dec) = self._gen_rnn(F.concat(gen_z, v_q, r, dim=1), [h_dec, c_dec])

                u = u + self._upsample_nn(h_dec)

            out = self._out_layer(u)

        return out, nd.stack(*qs, axis=0), nd.stack(*ps, axis=0)  # qs and ps: steps x batch x latent


def generate_samples(gqn_nn: GenerativeQueryNetwork, data_iter: GQNDataIter, context_size: int,
                     image_shape: Tuple[int, int, int],
                     save_path: str, save_prefix: str,
                     scale_factor: float = 1.):
    """
    Generate samples from GQN and save them to disk.

    :param gqn_nn: Trained GQN model.
    :param data_iter: Data iterator to sample the context (i.e., conditioning variables) from.
    :param context_size: Number of context images.
    :param image_shape: CxHxW of image.
    :param save_path: Path to save gif files in.
    :param save_prefix: Prefix for gif filenames.
    :param scale_factor: Scale images by this factor.
    :return:
    """
    orig_context_size = data_iter._context_size_range
    data_iter._context_size_range = (context_size,)
    batch = data_iter.next()
    data_iter._context_size_range = orig_context_size
    context_frames, context_cameras, query_cameras = batch.data[:3]
    query_frames = batch.label[0]
    samples = gqn_nn.generate(query_cameras, context_frames, context_cameras, include_intermediate=False)

    samples = samples.transpose((0, 2, 3, 1))  # from CxHxW to HxWxC
    samples = (samples.asnumpy() * 255).astype(np.uint8)
    context_frames = context_frames.transpose((0, 1, 3, 4, 2))
    context_frames = (context_frames.asnumpy() * 255).astype(np.uint8)
    query_frames = query_frames.transpose((0, 2, 3, 1))
    query_frames = (query_frames.asnumpy() * 255).astype(np.uint8)

    for i, sample in enumerate(samples):
        img = np.zeros((image_shape[1], image_shape[2]*(context_size+2), image_shape[0]), dtype=np.uint8)
        for j, cf in enumerate(context_frames[i]):
            img[:, j*image_shape[2]:(j+1)*image_shape[2], :] = cf
        img[:, -2*image_shape[2]:-image_shape[2], :] = query_frames[i]
        img[:, -image_shape[2]:, :] = sample

        file_path = os.path.join(save_path, "{}_{}.png".format(save_prefix, i))
        # rescale image
        scaled_sample = transform.rescale(img, scale=scale_factor, anti_aliasing=True, multichannel=True,
                                          preserve_range=True).astype(np.uint8)

        imageio.imwrite(file_path, scaled_sample)


def generate_sampling_gif(gqn_nn: GenerativeQueryNetwork, data_iter: GQNDataIter, context_size: int,
                          image_shape: Tuple[int, int, int],
                          save_path: str, save_prefix: str, scale_factor: float = 1.):
    """
    Generate animations of sampling from the given model.

    :param gqn_nn: Trained GQN model.
    :param data_iter: Data iterator to sample the context (i.e., conditioning variables) from.
    :param context_size: Number of context images.
    :param image_shape: CxHxW of image.
    :param save_path: Path to save gif files in.
    :param save_prefix: Prefix for gif filenames.
        attention.
    :param scale_factor: Scale images by this factor.
    :return:
    """
    orig_context_size = data_iter._context_size_range
    data_iter._context_size_range = (context_size,)
    batch = data_iter.next()
    data_iter._context_size_range = orig_context_size
    context_frames, context_cameras, query_cameras = batch.data[:3]
    query_frames = batch.label[0]
    samples = gqn_nn.generate(query_cameras, context_frames, context_cameras, include_intermediate=True)

    samples = samples.transpose((1, 0, 3, 4, 2))  # from CxHxW to HxWxC
    samples = (samples.asnumpy() * 255).astype(np.uint8)
    context_frames = context_frames.transpose((0, 1, 3, 4, 2))
    context_frames = (context_frames.asnumpy() * 255).astype(np.uint8)
    query_frames = query_frames.transpose((0, 2, 3, 1))
    query_frames = (query_frames.asnumpy() * 255).astype(np.uint8)

    for i, sample in enumerate(samples):
        scaled_samples = []
        file_path = os.path.join(save_path, "{}_{}.gif".format(save_prefix, i))
        for t, sample_t in enumerate(sample):
            img = np.zeros((image_shape[1], image_shape[2]*(context_size+2), image_shape[0]), dtype=np.uint8)
            for j, cf in enumerate(context_frames[i]):
                img[:, j*image_shape[2]:(j+1)*image_shape[2], :] = cf
            img[:, -2*image_shape[2]:-image_shape[2], :] = query_frames[i]
            img[:, -image_shape[2]:, :] = sample_t

            # rescale image
            scaled_samples.append(transform.rescale(img, scale=scale_factor, anti_aliasing=True, multichannel=True,
                                  preserve_range=True).astype(np.uint8))

        imageio.mimwrite(file_path, scaled_samples, duration=0.1)
