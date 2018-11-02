#!/usr/bin/python3

import argparse

import mxnet as mx
from mxnet.gluon import nn, Trainer
from mxnet.gluon.loss import SigmoidBinaryCrossEntropyLoss

from common import get_omniglot, PlotGradientHistogram, PlotGenerateImage, train
from convdraw.core import ConvDRAW, ConvDRAWLoss, generate_sampling_gif


def build_encoder_nn():
    net = nn.HybridSequential()
    net.add(nn.Conv2D(channels=16, kernel_size=(3, 3), strides=(2, 2), padding=(1, 1)))  # 14x14
    net.add(nn.Conv2D(channels=32, kernel_size=(3, 3), strides=(2, 2), padding=(1, 1)))  # 7x7
    net.add(nn.Conv2D(channels=32, kernel_size=(3, 3), strides=(2, 2), padding=(1, 1)))  # 4x4
    net.add(nn.Conv2D(channels=32, kernel_size=(3, 3), strides=(2, 2), padding=(1, 1)))  # 2x2
    return net, (32, 2, 2)


def build_decoder_nn():
    net = nn.HybridSequential()
    net.add(nn.Conv2DTranspose(channels=32, kernel_size=(3, 3), strides=(2, 2), padding=(1, 1)))  # 3x3
    net.add(nn.Conv2DTranspose(channels=32, kernel_size=(3, 3), strides=(2, 2)))  # 7x7
    net.add(nn.Conv2DTranspose(channels=16, kernel_size=(3, 3), strides=(2, 2), padding=(1, 1)))  # 13x13
    net.add(nn.Conv2DTranspose(channels=1, kernel_size=(4, 4), strides=(2, 2)))  # 28x28
    return net


def parse_args():
    ap = argparse.ArgumentParser(description="Train ConvDRAW on Omniglot dataset")
    ap.add_argument("--batch_size", '-b', type=int, default=128, help="Batch size")
    ap.add_argument("--num_steps", '-s', type=int, default=32, help="Number of recurrent steps")
    ap.add_argument("--latent_dim", '-l', type=int, default=16, help="Latent space dimension (number of elements)")
    ap.add_argument("--num_recurrent_maps", '-u', type=int, default=64, help="Number of feature maps in recurrent "
                                                                             "encoder and decoder")
    ap.add_argument("--learning_rate", '-r', type=float, default=1e-3, help="Learning rate")
    ap.add_argument("--num_train_samples", '-t', type=int, default=1e6, help="Number of training samples")
    ap.add_argument("--num_val_samples", '-v', type=int, default=256, help="Number of validation samples "
                                                                           "(per validation run)")
    ap.add_argument("--val_freq", '-f', type=int, default=1e4, help="Validation frequency (run validation every "
                                                                    "val_freq training samples)")
    ap.add_argument("--gpu", action='store_true', default=False, help="If True, train on GPU")
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
                            input_shape=input_shape, latent_dim=args.latent_dim,
                            encoder_output_shape=encoder_output_shape, rnn_hidden_channels=args.num_recurrent_maps,
                            kernel_size=(3, 3), ctx=ctx, prefix='conv_draw')

    model_params = conv_draw_nn.collect_params()
    model_params.initialize(ctx=ctx)

    # loss function
    loss_fn = ConvDRAWLoss(SigmoidBinaryCrossEntropyLoss(from_sigmoid=False, batch_axis=0), input_dim, args.latent_dim)

    # optimizer
    trainer = Trainer(params=model_params, optimizer='adam',
                      optimizer_params={'learning_rate': args.learning_rate, 'clip_gradient': 10.})

    # forward function for training
    def forward_fn(batch):
        x = batch.data[0].as_in_context(ctx)
        y, qs, ps = conv_draw_nn(x)
        loss = loss_fn(x, qs, ps, y)
        return loss

    plot_grad_cb = PlotGradientHistogram(conv_draw_nn, freq=500)
    generate_image_cb = PlotGenerateImage(conv_draw_nn, freq=args.val_freq, image_shape=input_shape)

    # train
    run_id = train(forward_fn, train_iter, val_iter, trainer,
                   args.num_train_samples, args.num_val_samples, args.val_freq,
                   args.logdir, (plot_grad_cb, generate_image_cb))

    # save model
    conv_draw_nn.save_parameters('results/conv_draw_{}.params'.format(run_id))

    # generate samples
    generate_sampling_gif(conv_draw_nn, image_shape=input_shape, save_path='results', save_prefix=run_id,
                          scale_factor=2.0)
