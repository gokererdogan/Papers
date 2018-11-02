"""
An MxNet implementation of the variational autoencoder described in
    Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes.
    arXiv:1312.6114

13 Aug 2018
goker erdogan
https://github.com/gokererdogan
"""
from typing import Optional, Tuple

import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
from mxnet.gluon import HybridBlock
from mxnet.gluon.loss import Loss
from mxnet.io import DataIter
from numpy import inf


class NormalSamplingBlock(HybridBlock):
    """
    This class implements a sampling layer that generates variables from a multivariate normal distribution with
    calculated mean and diagonal covariance matrix.
    This layer is used in the variational autoencoder for sampling the latent variables.
    """
    def __init__(self, batch_size: int, latent_dim: int, prefix: Optional[str] = None):
        """
        :param batch_size: Batch size.
        :param latent_dim: Dimensionality of latent space.
        :param prefix: Prefix for block.
        """
        super().__init__(prefix=prefix)

        self._batch_size = batch_size
        self._latent_dim = latent_dim

    def hybrid_forward(self, F, x, *args, **kwargs):
        # Sample z from Normal(z|mu, sigma) using the reparameterization trick.
        eps = F.random.normal(shape=(self._batch_size, self._latent_dim))  # draw random normal values
        z = x[:, 0:self._latent_dim] + (eps * F.exp(x[:, self._latent_dim:]))  # calculate latent variables
        return z


class VAELossKLTerm(Loss):
    """
    KL term in variational lower bound used for training variational autoencoder.

    First term in Eqn. 10 in the paper.
    Note we return the positive KL so we would like to minimize the sum of this with the fit term.
    """
    def __init__(self, latent_dim: int):
        """
        :param latent_dim: Dimensionality of latent space.
        """
        super().__init__(weight=1, batch_axis=0)

        self._latent_dim = latent_dim

    def hybrid_forward(self, F, q, *args, **kwargs):
        # q is batches x latent space * 2 [mu, log_sd]
        mu = q[:, 0:self._latent_dim]
        log_sd = q[:, self._latent_dim:]

        # first term in Eq. 10. acts as a regularizer
        kl_term = 0.5 * (-self._latent_dim - F.sum(2. * log_sd, axis=0, exclude=True) +
                         F.sum(F.square(mu), axis=0, exclude=True) +
                         F.sum(F.exp(2. * log_sd), axis=0, exclude=True))

        return kl_term


class VAELoss(Loss):
    """
    Negative variational lower bound used for training variational autoencoder.

    See Eq. 10 in the paper.
    """
    def __init__(self, fit_loss: Loss, input_dim: int, latent_dim: int):
        """
        :param fit_loss: Loss used for p(x|z), i.e., reconstruction loss. Note the output of VAE is not passed through
            sigmoid. We assume that the loss function averages over (input/output) features, rather than summing.
        :param input_dim: Input dimensionality.
        :param latent_dim: Dimensionality of latent space
        """
        super().__init__(weight=1, batch_axis=0)

        self._input_dim = input_dim
        self._latent_dim = latent_dim

        with self.name_scope():
            self._fit_loss = fit_loss
            self._kl_loss = VAELossKLTerm(latent_dim)

    def hybrid_forward(self, F, x, *args, **kwargs):
        # x: input, args[0]: latent prob. distribution, args[1]: output of VAE (before sigmoid)
        q = args[0]
        y = args[1]

        kl_term = self._kl_loss(q)
        fit_term = self._fit_loss(y, x) * self._input_dim  # fix averaging over features by mult by input dim

        return fit_term + kl_term


class VAE(HybridBlock):
    """
    Variational Autoencoder (VAE) class
    """
    def __init__(self, enc_nn: HybridBlock, dec_nn: HybridBlock, batch_size: int, latent_dim: int,
                 prefix: Optional[str] = None):
        """
        :param enc_nn: Encoder network. Note the number of output units must be 2 * latent_dim. The network should
            output real values (in [-inf, +inf]), unless you have a good reason to constrain the domain of latent
            variables.
        :param dec_nn: Decoder network. Note the number of output units must be the size of input. Do not pass the
            output through sigmoid. This is done by this class when needed.
        :param batch_size: Batch size.
        :param latent_dim: Dimensionality of latent space.
        :param prefix: Prefix for block.
        """
        super().__init__(prefix=prefix)

        self._batch_size = batch_size
        self._latent_dim = latent_dim

        with self.name_scope():
            self._enc_nn = enc_nn
            self._dec_nn = dec_nn
            self._latent_layer = NormalSamplingBlock(self._batch_size, self._latent_dim)

    @property
    def latent_dim(self):
        return self._latent_dim

    def hybrid_forward(self, F, x, *args, **kwargs):
        q = self._enc_nn(x)
        z = self._latent_layer(q)
        y = self._dec_nn(z)

        return y, q

    def decode(self, z):
        """
        Given a latent vector, predict the input. z -> x
        :param z: Latent matrix (batch x features).
        :return:
        """
        return nd.sigmoid(self._dec_nn(z))

    def encode(self, x):
        """
        Given input, calculate the latent variable distribution.
        :param x: Input matrix (batch x features).
        :return: q (batch x [mu, log_sd])
        """
        return self._enc_nn(x)


def generate_2d_latent_space_image(nn: VAE, data_iter: DataIter, input_shape: Tuple[int, int], n: int = 20,
                                   ctx: mx.Context = mx.cpu()):
    """
    Given a trained generative model, plot the 2D latent space. Works only for models with 2D latent spaces.

    :param nn: Generative model.
    :param data_iter: Data iterator to calculate the min, max values of latent variables.
    :param input_shape: Input shape (HxW).
    :param n: Number of images in rows/cols. Produces n x n array of images.
    :param ctx: MxNet context.
    :return: A single image of n x n array of images.
    """
    assert nn.latent_dim
    h, w = input_shape
    img = np.zeros((h*n, w*n))

    # calculate min and max latent variable values over dataset
    minz = np.array([+inf, +inf])
    maxz = np.array([-inf, -inf])

    for batch in data_iter:
        q = nn.encode(batch.data[0].as_in_context(ctx))
        mu = q[:, 0:2].asnumpy()

        min = np.min(mu, axis=0)
        max = np.max(mu, axis=0)
        minz[min < minz] = min[min < minz]
        maxz[max > maxz] = max[max > maxz]

    # predict input for each latent and write to image
    z1_vals = np.linspace(minz[0], maxz[0], n)
    z2_vals = np.linspace(minz[1], maxz[1], n)
    for i, z1 in enumerate(z1_vals):
        for j, z2 in enumerate(z2_vals):
            xp = nn.decode(nd.array([[z1, z2]], ctx=ctx))
            img[(i*h):((i+1)*h), (j*w):((j+1)*w)] = xp.asnumpy().reshape(input_shape)

    return img
