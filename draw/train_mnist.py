import argparse
from time import strftime

import mxboard
import mxnet as mx
import mxnet.ndarray as nd
import tqdm
from mxnet import autograd
from mxnet.gluon import Trainer
from mxnet.gluon.loss import SigmoidBinaryCrossEntropyLoss
from mxnet.io import DataIter
from numpy import inf

from common import get_mnist
from draw.core import DRAW, DRAWLoss, NoAttentionRead, NoAttentionWrite, SelectiveAttentionRead, SelectiveAttentionWrite


def parse_args():
    ap = argparse.ArgumentParser(description="Train DRAW on MNIST dataset")
    ap.add_argument("--batch_size", '-b', type=int, default=128, help="Batch size")
    ap.add_argument("--input_height", '-hh', type=int, default=28, help="Height of input image")
    ap.add_argument("--input_width", '-ww', type=int, default=28, help="Height of input image")
    ap.add_argument("--num_steps", '-s', type=int, default=8, help="Height of input image")
    ap.add_argument("--num_recurrent_units", '-u', type=int, default=256, help="Number of units in recurrent encoder "
                                                                               "and decoder")
    ap.add_argument("--latent_dim", '-l', type=int, default=100, help="Latent space dimension (number of elements)")
    ap.add_argument("--learning_rate", '-r', type=float, default=1e-3, help="Learning rate")
    ap.add_argument("--num_train_samples", '-t', type=int, default=1e5, help="Number of training samples")
    ap.add_argument("--num_val_samples", '-v', type=int, default=256, help="Number of validation samples "
                                                                           "(per validation run)")
    ap.add_argument("--val_freq", '-f', type=int, default=2e3, help="Validation frequency (run validation every "
                                                                    "val_freq training samples)")
    ap.add_argument("--gpu", action='store_true', default=False, help="If True, train on GPU")
    ap.add_argument("--attention", action='store_true', default=False, help="If True, train with selective attention.")
    ap.add_argument("--read_size", '-ar', type=int, default=2, help="If True, train with selective attention.")
    ap.add_argument("--write_size", '-aw', type=int, default=5, help="If True, train with selective attention.")

    return ap.parse_args()


# TODO: refactor training code
def _get_batch(data_iter: DataIter):
    """
    Helper function for getting a batch from a data iterator continuously.

    :param data_iter: Data iterator.
    :return: Data batch.
    """
    try:
        batch = data_iter.next()
    except StopIteration:
        data_iter.reset()
        batch = data_iter.next()

    return batch


if __name__ == "__main__":
    args = parse_args()

    ctx = mx.gpu() if args.gpu else mx.cpu()
    input_shape = (args.input_height, args.input_width)
    input_dim = args.input_height * args.input_width

    train_iter, val_iter = get_mnist(batch_size=args.batch_size, input_shape=input_shape)

    # build the network
    if not args.attention:
        read_nn = NoAttentionRead()
        write_nn = NoAttentionWrite(input_dim)
    else:
        read_nn = SelectiveAttentionRead(args.read_size, input_shape, args.batch_size)
        write_nn = SelectiveAttentionWrite(args.read_size, input_shape, args.batch_size)

    draw_nn = DRAW(read_nn, write_nn, args.num_steps, args.batch_size, args.num_recurrent_units, input_dim,
                   args.latent_dim)
    model_params = draw_nn.collect_params()
    model_params.initialize(ctx=ctx)

    # loss function
    loss_fn = DRAWLoss(SigmoidBinaryCrossEntropyLoss(from_sigmoid=False, batch_axis=0), input_dim, args.latent_dim)

    # optimizer
    trainer = Trainer(params=model_params, optimizer='adam',
                      optimizer_params={'learning_rate': args.learning_rate, 'clip_gradient': 10.})

    # TODO: refactor training code
    # Training/validation loop
    sw = mxboard.SummaryWriter(logdir='results')
    pm = tqdm.tqdm(total=args.num_train_samples)

    remaining = args.num_train_samples
    last_train_loss = inf
    last_val_loss = inf
    while remaining > 0:
        batch = _get_batch(train_iter)
        x = batch.data[0].as_in_context(ctx)

        # train step
        with autograd.record():
            y, qs = draw_nn(x)
            loss = loss_fn(x, qs, y)
        autograd.backward(loss)

        batch_size = loss.shape[0]
        trainer.step(batch_size=batch_size)

        remaining -= batch_size
        last_train_loss = nd.mean(loss).asscalar()  # loss per sample
        # plot loss
        sw.add_scalar('Loss', {'Training': last_train_loss}, args.num_train_samples - remaining)
        if (args.num_train_samples - remaining) % 250 < batch_size:
            for k, p in model_params.items():
                g = p.grad().asnumpy()
                sw.add_histogram(k, g, args.num_train_samples - remaining, bins=10)

        pm.update(n=batch_size)

        # validation step
        if (args.num_train_samples - remaining) % args.val_freq < batch_size:
            tot_val_loss = 0.0
            j = args.num_val_samples
            while j > 0:
                batch = _get_batch(val_iter)
                x = batch.data[0].as_in_context(ctx)

                # validation step
                y, q = draw_nn(x)
                loss = loss_fn(x, q, y)
                tot_val_loss += nd.sum(loss).asscalar()

                j -= loss.shape[0]

            last_val_loss = tot_val_loss / (args.num_val_samples - j)  # loss per sample
            sw.add_scalar('Loss', {'Validation': last_val_loss}, args.num_train_samples - remaining)

            # generate image from model
            img = draw_nn.generate().asnumpy()[0].reshape(input_shape)
            sw.add_image('Generated_image', img, args.num_train_samples - remaining)

            sw.flush()

        pm.set_postfix({'Train loss': last_train_loss, 'Val loss': last_val_loss})

    pm.close()
    sw.flush()
    sw.close()

    # save model
    draw_nn.save_parameters('results/draw_{}.params'.format(strftime('%Y%M%d%H%m%S')))

