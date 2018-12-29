#!/usr/bin/python3

import argparse

import mxnet as mx
import numpy as np
from mxnet.gluon import nn, Trainer
from mxnet.gluon.nn import Conv2D, Conv2DTranspose

from common import PlotGradientHistogram, PlotGenerateImage, train, WithELU, GQNDataIter
from gqn.core import RepresentationNetworkTower, GenerativeQueryNetwork, NormalFitLoss, GQNLoss, PixelStdScheduler


class GQNLearningRateSchedule:
    """
    Learning rate scheduler for GQN.

    See Table S1 in supplementary materials.
    """
    def __init__(self, lr_initial: float, lr_final: float, num_train_samples: int, batch_size: int):
        self._lr_initial = lr_initial
        self._lr_final = lr_final
        self._num_train_samples = num_train_samples
        self._batch_size = batch_size

    def __call__(self, iteration: int):
        samples_processed = (iteration - 1)*self._batch_size
        remain_ratio = 1 - (samples_processed / float(self._num_train_samples))
        lr = max(self._lr_final, self._lr_final + (self._lr_initial - self._lr_final)*remain_ratio)
        return lr


def build_downsample_nn():
    nnet = nn.HybridSequential()
    nnet.add(WithELU(Conv2D(channels=32, kernel_size=(4, 4), strides=(4, 4))))
    return nnet, (32, 16, 16)


def build_upsample_nn():
    nnet = nn.HybridSequential()
    nnet.add(WithELU(Conv2DTranspose(channels=32, kernel_size=(4, 4), strides=(4, 4))))
    return nnet, (32, 64, 64)


def parse_args():
    ap = argparse.ArgumentParser(description="Train Generative Query Network")
    ap.add_argument("--dataset_name", '-d', type=str, default='shepard_metzler_5_parts', help="Dataset name")
    ap.add_argument("--max_context_size", '-c', type=int, default=12, help="Maximum number of views in context")
    ap.add_argument("--batch_size", '-b', type=int, default=32, help="Batch size")
    ap.add_argument("--num_steps", '-s', type=int, default=12, help="Number of recurrent steps")
    ap.add_argument("--num_latent_maps", '-l', type=int, default=3, help="Number of latent feature maps")
    ap.add_argument("--num_recurrent_maps", '-u', type=int, default=64, help="Number of feature maps in recurrent "
                                                                             "encoder and decoder")
    ap.add_argument("--pixel_std_initial", '-pi', type=float, default=2.0, help="Initial pixel standard deviation")
    ap.add_argument("--pixel_std_final", '-pf', type=float, default=0.7, help="Final pixel standard deviation")
    ap.add_argument("--learning_rate_initial", '-ri', type=float, default=5*1e-4, help="Initial learning rate")
    ap.add_argument("--learning_rate_final", '-rf', type=float, default=4*1e-4, help="Final learning rate")
    ap.add_argument("--num_train_samples", '-t', type=int, default=5e6, help="Number of training samples")
    ap.add_argument("--num_val_samples", '-v', type=int, default=128, help="Number of validation samples "
                                                                           "(per validation run)")
    ap.add_argument("--val_freq", '-f', type=int, default=2e4, help="Validation frequency (run validation every "
                                                                    "val_freq training samples)")
    ap.add_argument("--run_suffix", type=str, default='', help='Suffix for run id.')
    ap.add_argument("--gpu", action='store_true', help="If True, train on GPU")
    ap.add_argument("--logdir", type=str, default='results', help='Log directory for mxboard')

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    ctx = mx.gpu() if args.gpu else mx.cpu()
    input_shape = (3, 64, 64)
    input_dim = int(np.prod(input_shape))

    context_size_range = tuple(range(1, args.max_context_size+1))
    train_iter = GQNDataIter(args.dataset_name, 'train', context_size_range=context_size_range,
                             batch_size=args.batch_size, ctx=ctx)
    val_iter = GQNDataIter(args.dataset_name, 'test', context_size_range=context_size_range,
                           batch_size=args.batch_size, ctx=ctx)

    rep_nn = RepresentationNetworkTower(args.batch_size)
    representation_shape = rep_nn.get_output_shape()[1:]
    downsample_nn, downsample_output_shape = build_downsample_nn()
    upsample_nn, upsample_output_shape = build_upsample_nn()

    gqn_nn = GenerativeQueryNetwork(rep_nn, downsample_nn, upsample_nn,
                                    num_steps=args.num_steps, batch_size=args.batch_size,
                                    input_shape=input_shape, num_latent_maps=args.num_latent_maps,
                                    downsample_output_shape=downsample_output_shape,
                                    upsample_output_shape=upsample_output_shape,
                                    representation_shape=representation_shape,
                                    rnn_hidden_channels=args.num_recurrent_maps,
                                    kernel_size=(5, 5), ctx=ctx, prefix='gqn')
    model_params = gqn_nn.collect_params()
    model_params.initialize(ctx=ctx)

    # loss function
    latent_shape = (args.num_latent_maps, *downsample_output_shape[1:])
    pixel_std_scheduler = PixelStdScheduler(std_initial=args.pixel_std_initial, std_final=args.pixel_std_final,
                                            num_train_samples=args.num_train_samples / 10)
    loss_fit = NormalFitLoss(std=args.pixel_std_initial)
    loss_fn = GQNLoss(loss_fit, input_dim, latent_shape)

    # optimizer
    lr_scheduler = GQNLearningRateSchedule(lr_initial=args.learning_rate_initial, lr_final=args.learning_rate_final,
                                           num_train_samples=args.num_train_samples*0.8, batch_size=args.batch_size)
    trainer = Trainer(params=model_params, optimizer='adam',
                      optimizer_params={'learning_rate': args.learning_rate_initial,
                                        'lr_scheduler': lr_scheduler})

    # forward function for training
    def forward_fn(batch, pixel_std):
        context_frames, context_cameras, query_cameras = batch.data[:3]
        query_frames = batch.label[0]
        y, qs, ps = gqn_nn(query_frames, query_cameras, context_frames, context_cameras)

        # update pixel std
        loss_fit.set_std(pixel_std)
        loss = loss_fn(query_frames, qs, ps, y)
        return loss

    plot_grad_cb = PlotGradientHistogram(gqn_nn, freq=1000)
    plot_batch = val_iter.next()
    plt_context_frames, plt_context_cameras, plt_query_cameras = plot_batch.data[:3]
    conditioning_variables=(plt_query_cameras, plt_context_frames, plt_context_cameras)
    generate_image_cb = PlotGenerateImage(gqn_nn, conditioning_variables=conditioning_variables,
                                          freq=args.val_freq, image_shape=input_shape)

    # train
    run_id = train(forward_fn, train_iter, val_iter, trainer, args.num_train_samples,
                   args.num_val_samples, args.val_freq, args.logdir,
                   validate_at_end=False,
                   hyperparam_scheduler=pixel_std_scheduler,
                   plot_callbacks=(plot_grad_cb, generate_image_cb), run_suffix=args.run_suffix)

    # save model
    gqn_nn.save_parameters('results/gqn_{}_{}.params'.format(args.dataset_name, run_id))

    # generate samples
    # generate_samples(conv_draw_nn, input_shape, 'results', run_id, scale_factor=2.0)
