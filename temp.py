import os

import mxnet as mx
import numpy as np

import imageio
from skimage import transform

from common import get_omniglot
from convdraw.core import ConvDRAW, generate_sampling_gif, generate_samples
from convdraw.train_omniglot import parse_args, build_decoder_nn, build_encoder_nn

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
    run_id = 'conv_draw_20181215211425'
    conv_draw_nn.load_parameters('results/{}.params'.format(run_id))

    # conv_draw_nn(train_iter.next().data[0])

    mx.random.seed(np.random.randint(1000000))
    generate_samples(conv_draw_nn, input_shape, 'results', run_id, scale_factor=2.0)
    # generate_sampling_gif(conv_draw_nn, input_shape, 'results', run_id, scale_factor=2.0)
