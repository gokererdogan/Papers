#!/usr/bin/python3

import argparse

import mxnet as mx
from mxnet.gluon import nn, Trainer, HybridBlock
from mxnet.gluon.loss import SigmoidBinaryCrossEntropyLoss
from mxnet.gluon.nn import Conv2D

from common import get_omniglot, PlotGradientHistogram, PlotGenerateImage, train
from convdraw.core import ConvDRAW, ConvDRAWLoss, generate_samples, Conv2DWithBatchNorm, Conv2DTransposeWithBatchNorm


class IdentityBlock(HybridBlock):
    def __init__(self, prefix: str = None):
        super().__init__(prefix=prefix)

    def hybrid_forward(self, F, x, *args, **kwargs):
        return x


class WithELU(HybridBlock):
    def __init__(self, block: HybridBlock, prefix: str = None):
        super().__init__(prefix=prefix)

        self._block = block

    def hybrid_forward(self, F, x, *args, **kwargs):
        y = self._block(x)
        return F.LeakyReLU(y, act_type='elu')


def build_encoder_nn():
    return IdentityBlock(), (1, 28, 28)


def build_decoder_nn():
    return Conv2D(channels=1, kernel_size=1)


def parse_args():
    ap = argparse.ArgumentParser(description="Train ConvDRAW on Omniglot dataset")
    ap.add_argument("--batch_size", '-b', type=int, default=8, help="Batch size")
    ap.add_argument("--num_steps", '-s', type=int, default=16, help="Number of recurrent steps")
    ap.add_argument("--num_latent_maps", '-l', type=int, default=3, help="Number of latent feature maps.")
    ap.add_argument("--num_recurrent_maps", '-u', type=int, default=64, help="Number of feature maps in recurrent "
                                                                             "encoder and decoder")
    ap.add_argument("--learning_rate", '-r', type=float, default=5*1e-4, help="Learning rate")
    ap.add_argument("--num_train_samples", '-t', type=int, default=1e5, help="Number of training samples")
    ap.add_argument("--num_val_samples", '-v', type=int, default=128, help="Number of validation samples "
                                                                           "(per validation run)")
    ap.add_argument("--val_freq", '-f', type=int, default=1e4, help="Validation frequency (run validation every "
                                                                    "val_freq training samples)")
    ap.add_argument("--gpu", action='store_true', help="If True, train on GPU")
    ap.add_argument("--logdir", type=str, default='results', help='Log directory for mxboard.')

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    ctx = mx.gpu() if args.gpu else mx.cpu()
    input_shape = (1, 28, 28)
    input_dim = input_shape[1]*input_shape[2]

    train_iter, val_iter = get_omniglot(batch_size=args.batch_size)

    encoder_nn, encoder_output_shape = build_encoder_nn()
    decoder_nn = build_decoder_nn()
    conv_draw_nn = ConvDRAW(encoder_nn, decoder_nn, num_steps=args.num_steps, batch_size=args.batch_size,
                            input_shape=input_shape, num_latent_maps=args.num_latent_maps,
                            encoder_output_shape=encoder_output_shape, rnn_hidden_channels=args.num_recurrent_maps,
                            kernel_size=(5, 5), ctx=ctx, prefix='conv_draw')

    model_params = conv_draw_nn.collect_params()
    model_params.initialize(ctx=ctx)

    # loss function
    latent_shape = (args.num_latent_maps, *encoder_output_shape[1:])
    loss_fn = ConvDRAWLoss(SigmoidBinaryCrossEntropyLoss(from_sigmoid=False, batch_axis=0), input_dim, latent_shape)

    # optimizer
    trainer = Trainer(params=model_params, optimizer='adam',
                      optimizer_params={'learning_rate': args.learning_rate, 'clip_gradient': 10.})

    # forward function for training
    def forward_fn(batch):
        x = batch.data[0].as_in_context(ctx)
        y, qs, ps = conv_draw_nn(x)
        loss = loss_fn(x, qs, ps, y)
        return loss

    plot_grad_cb = PlotGradientHistogram(conv_draw_nn, freq=1000)
    generate_image_cb = PlotGenerateImage(conv_draw_nn, freq=args.val_freq, image_shape=input_shape)

    # train
    run_id = train(forward_fn, train_iter, val_iter, trainer,
                   args.num_train_samples, args.num_val_samples, args.val_freq,
                   args.logdir, (plot_grad_cb, generate_image_cb))

    # save model
    conv_draw_nn.save_parameters('results/conv_draw_{}.params'.format(run_id))

    # generate samples
    generate_samples(conv_draw_nn, input_shape, 'results', run_id, scale_factor=2.0)
