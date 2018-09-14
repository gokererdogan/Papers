import argparse

import mxnet as mx
from mxnet.gluon import Trainer
from mxnet.gluon.loss import SigmoidBinaryCrossEntropyLoss

from common import get_binarized_mnist, train, PlotGradientHistogram, PlotGenerateImage
from draw.core import DRAW, DRAWLoss, NoAttentionRead, NoAttentionWrite, SelectiveAttentionRead, \
    SelectiveAttentionWrite, generate_sampling_gif


def parse_args():
    ap = argparse.ArgumentParser(description="Train DRAW on MNIST dataset")
    ap.add_argument("--batch_size", '-b', type=int, default=128, help="Batch size")
    ap.add_argument("--input_height", '-hh', type=int, default=28, help="Height of input image")
    ap.add_argument("--input_width", '-ww', type=int, default=28, help="Height of input image")
    ap.add_argument("--num_steps", '-s', type=int, default=64, help="Height of input image")
    ap.add_argument("--num_recurrent_units", '-u', type=int, default=256, help="Number of units in recurrent encoder "
                                                                               "and decoder")
    ap.add_argument("--latent_dim", '-l', type=int, default=10, help="Latent space dimension (number of elements)")
    ap.add_argument("--learning_rate", '-r', type=float, default=1e-3, help="Learning rate")
    ap.add_argument("--num_train_samples", '-t', type=int, default=1e6, help="Number of training samples")
    ap.add_argument("--num_val_samples", '-v', type=int, default=256, help="Number of validation samples "
                                                                           "(per validation run)")
    ap.add_argument("--val_freq", '-f', type=int, default=1e4, help="Validation frequency (run validation every "
                                                                    "val_freq training samples)")
    ap.add_argument("--gpu", action='store_true', default=False, help="If True, train on GPU")
    ap.add_argument("--attention", action='store_true', default=False, help="If True, train with selective attention.")
    ap.add_argument("--read_size", '-ar', type=int, default=4, help="If True, train with selective attention.")
    ap.add_argument("--write_size", '-aw', type=int, default=5, help="If True, train with selective attention.")
    ap.add_argument("--logdir", type=str, default='results', help='Log directory for mxboard.')

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    ctx = mx.gpu() if args.gpu else mx.cpu()
    input_shape = (args.input_height, args.input_width)
    input_dim = args.input_height * args.input_width

    train_iter, val_iter = get_binarized_mnist(batch_size=args.batch_size, input_shape=input_shape)

    # build the network
    if not args.attention:
        read_nn = NoAttentionRead()
        write_nn = NoAttentionWrite(input_dim)
    else:
        read_nn = SelectiveAttentionRead(args.read_size, input_shape, args.batch_size)
        write_nn = SelectiveAttentionWrite(args.write_size, input_shape, args.batch_size)

    draw_nn = DRAW(read_nn, write_nn, args.num_steps, args.batch_size, args.num_recurrent_units, input_dim,
                   args.latent_dim, ctx)
    model_params = draw_nn.collect_params()
    model_params.initialize(ctx=ctx)

    # loss function
    loss_fn = DRAWLoss(SigmoidBinaryCrossEntropyLoss(from_sigmoid=False, batch_axis=0), input_dim, args.latent_dim)

    # optimizer
    trainer = Trainer(params=model_params, optimizer='adam',
                      optimizer_params={'learning_rate': args.learning_rate, 'clip_gradient': 10.})

    # forward function for training
    def forward_fn(batch):
        x = batch.data[0].as_in_context(ctx)
        y, qs = draw_nn(x)
        loss = loss_fn(x, qs, y)
        return loss

    plot_grad_cb = PlotGradientHistogram(draw_nn, freq=500)
    generate_image_cb = PlotGenerateImage(draw_nn, freq=args.val_freq, image_shape=input_shape)

    # train
    run_id = train(forward_fn, train_iter, val_iter, trainer,
                   args.num_train_samples, args.num_val_samples, args.val_freq,
                   args.logdir, (plot_grad_cb, generate_image_cb))

    # save model
    draw_nn.save_parameters('results/draw_{}.params'.format(run_id))

    # generate samples
    generate_sampling_gif(draw_nn, image_shape=input_shape, save_path='results', save_prefix=run_id,
                          draw_attention=args.attention, scale_factor=2.0)

