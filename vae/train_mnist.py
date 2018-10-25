#!/usr/bin/python3

import argparse
import os

import mxboard
import mxnet as mx
import mxnet.gluon.nn as nn
from mxnet.gluon import Trainer
from mxnet.gluon.loss import SigmoidBinaryCrossEntropyLoss

from common import get_binarized_mnist, train
from vae.core import generate_2d_latent_space_image, VAE, VAELoss


def parse_args():
    ap = argparse.ArgumentParser(description="Train a variational autoencoder on MNIST dataset")
    ap.add_argument("--batch_size", '-b', type=int, default=128, help="Batch size")
    ap.add_argument("--input_height", '-hh', type=int, default=28, help="Height of input image")
    ap.add_argument("--input_width", '-ww', type=int, default=28, help="Height of input image")
    ap.add_argument("--num_encoder_units", '-e', type=int, default=512, help="Number of units in encoder hidden layer")
    ap.add_argument("--num_decoder_units", '-d', type=int, default=512, help="Number of units in decoder hidden layer")
    ap.add_argument("--latent_dim", '-l', type=int, default=2, help="Latent space dimension (number of elements)")
    ap.add_argument("--learning_rate", '-r', type=float, default=1e-4, help="Learning rate")
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
    input_shape = (args.input_height, args.input_width)
    input_dim = args.input_height * args.input_width

    train_iter, val_iter = get_binarized_mnist(batch_size=args.batch_size, input_shape=input_shape)

    # build the network
    enc_nn = nn.HybridSequential()
    enc_nn.add(nn.Dense(units=args.num_encoder_units, activation='relu'))
    enc_nn.add(nn.Dense(units=args.latent_dim * 2))

    dec_nn = nn.HybridSequential()
    dec_nn.add(nn.Dense(units=args.num_decoder_units, activation='relu'))
    dec_nn.add(nn.Dense(units=input_dim))

    vae_nn = VAE(enc_nn, dec_nn, args.batch_size, args.latent_dim)
    model_params = vae_nn.collect_params()
    model_params.initialize(ctx=ctx)

    # loss function
    loss_fn = VAELoss(SigmoidBinaryCrossEntropyLoss(from_sigmoid=False, batch_axis=0), input_dim, args.latent_dim)

    # optimizer
    trainer = Trainer(params=model_params, optimizer='adam',
                      optimizer_params={'learning_rate': args.learning_rate})

    # forward function for training
    def forward_fn(batch):
        x = batch.data[0].as_in_context(ctx)
        y, q = vae_nn(x)
        loss = loss_fn(x, q, y)
        return loss

    # train
    run_id = train(forward_fn, train_iter, val_iter, trainer,
                   args.num_train_samples, args.num_val_samples, args.val_freq,
                   args.logdir)

    # generate latent space figure if latent dim = 2
    sw = mxboard.SummaryWriter(logdir=os.path.join(args.logdir, run_id))
    if args.latent_dim == 2:
        img = generate_2d_latent_space_image(vae_nn, val_iter, input_shape, n=20, ctx=ctx)
        sw.add_image('2D_Latent_space', img)
    sw.close()

