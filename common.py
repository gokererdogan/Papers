import pickle
from time import strftime
from typing import List, Callable, Tuple, Union
import os
from pathlib import Path

import mxboard
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
import tqdm
from mxnet import autograd
from mxnet.gluon import Parameter, Trainer, HybridBlock
from mxnet.gluon.data import BatchSampler, RandomSampler
from mxnet.test_utils import get_mnist_ubyte

DATA_FOLDER_PATH = Path(__file__).parent / 'data'


class BinarizedMNISTIter(mx.io.DataIter):
    def __init__(self, image, label, input_shape, batch_size, flat):
        super().__init__(batch_size=batch_size)

        self._mnist_iter = mx.io.MNISTIter(image=image, label=label, input_shape=input_shape, batch_size=batch_size,
                                           flat=flat, num_parts=1, part_index=0)

    def next(self):
        batch = self._mnist_iter.next()
        # binarize
        batch.data[0] = batch.data[0] > 0.5
        return batch

    def reset(self):
        self._mnist_iter.reset()


class GQNDataIter(mx.io.DataIter):
    """
    Data iterator for GQN datasets.

    Batch format:
        batch.data[0]: Context image frames, shape: Batch x Sequence x 3 x H x w
        batch.data[1]: Context camera positions, shape: Batch x Sequence x Camera feats
        batch.data[2]: Query camera position, shape: Batch x Camera feats
        batch.label[0]: Query image frame, shape: Batch x 3 x H x W

    See data/gqn_data_to_recordio.py for how to convert GQN tfrecord files to an MXNet recordio file.
    """
    def __init__(self, dataset_name: str, dataset_type: str, context_size_range: Tuple[int], batch_size: int,
                 ctx: mx.Context):
        """
        :param dataset_name: Dataset name.
        :param dataset_type: 'train' or 'test'.
        :param context_size_range: Range of context size (number of images taken from a scene, except the query image).
        :param batch_size: Minibatch size.
        :param ctx: Context for data (cpu or gpu).
        """
        super().__init__(batch_size=batch_size)
        idx_filepath = str(Path(DATA_FOLDER_PATH, '{}_{}.idx'.format(dataset_name, dataset_type)))
        rec_filepath = str(Path(DATA_FOLDER_PATH, '{}_{}.rec'.format(dataset_name, dataset_type)))
        self._rec_file = mx.recordio.MXIndexedRecordIO(idx_filepath, rec_filepath, 'r', key_type=int)

        self._dataset_name = dataset_name
        self._dataset_type = dataset_type
        self._context_size_range = context_size_range
        self._batch_size = batch_size
        self._ctx = ctx

        self._num_scenes = len(self._rec_file.keys)
        self._batch_sampler = BatchSampler(RandomSampler(self._num_scenes), self._batch_size, last_batch='rollover')
        self.reset()

    def _read_single_record(self, i, context_size):
        recs = np.random.choice(pickle.loads(self._rec_file.read_idx(i)), size=context_size+1, replace=False)
        imgs = [mx.recordio.unpack_img(rec) for rec in recs]

        cameras = np.stack([img[0].label for img in imgs], axis=0)
        pos = cameras[:, 0:3]
        yaw = cameras[:, 3:4]
        pitch = cameras[:, 4:5]
        cameras = np.hstack((pos, np.sin(yaw), np.cos(yaw), np.sin(pitch), np.cos(pitch)))

        frames = [img[1].transpose((2, 0, 1)) / 255. for img in imgs]  # normalize to 0-1

        context_cameras = np.stack(cameras[0:context_size], axis=0)
        context_frames = np.stack(frames[0:context_size], axis=0)
        query_camera = cameras[-1]
        query_frame = frames[-1]

        return context_frames, context_cameras, query_frame, query_camera

    def _to_batch(self, records):
        batch_context_frames = nd.array(np.stack([rec[0] for rec in records], axis=0), ctx=self._ctx)
        batch_context_cameras = nd.array(np.stack([rec[1] for rec in records], axis=0), ctx=self._ctx)
        batch_query_frames = nd.array(np.stack([rec[2] for rec in records], axis=0), ctx=self._ctx)
        batch_query_cameras = nd.array(np.stack([rec[3] for rec in records], axis=0), ctx=self._ctx)
        batch = mx.io.DataBatch(data=[batch_context_frames, batch_context_cameras, batch_query_cameras],
                                label=[batch_query_frames])
        return batch

    def next(self):
        batch_ix = self._batch_sampler_iter.__next__()

        context_size = np.random.choice(self._context_size_range)
        records = [self._read_single_record(i, context_size) for i in batch_ix]
        batch = self._to_batch(records)

        return batch

    def reset(self):
        # get a new iterator from batch sampler to reset
        self._batch_sampler_iter = self._batch_sampler.__iter__()


get_mnist = mx.test_utils.get_mnist_iterator


def get_binarized_mnist(batch_size, input_shape):
    """
    Returns training and validation iterators for binarized MNIST dataset
    """
    get_mnist_ubyte()
    flat = False if len(input_shape) == 3 else True

    train_dataiter = BinarizedMNISTIter(
        image=str(DATA_FOLDER_PATH / "train-images-idx3-ubyte"),
        label=str(DATA_FOLDER_PATH / "train-labels-idx1-ubyte"),
        input_shape=input_shape,
        batch_size=batch_size,
        flat=flat)

    val_dataiter = BinarizedMNISTIter(
        image=str(DATA_FOLDER_PATH / "t10k-images-idx3-ubyte"),
        label=str(DATA_FOLDER_PATH / "t10k-labels-idx1-ubyte"),
        input_shape=input_shape,
        batch_size=batch_size,
        flat=flat)

    return train_dataiter, val_dataiter


def get_omniglot(batch_size):
    """
    Returns training and validation data iterators for the Omniglot dataset.
    """
    train_file = Path(DATA_FOLDER_PATH / 'omniglot_train_img.npy')
    train_dataiter = mx.io.NDArrayIter(np.load(str(train_file)), batch_size=batch_size, shuffle=True)

    val_file = Path(DATA_FOLDER_PATH / 'omniglot_test_img.npy')
    val_dataiter = mx.io.NDArrayIter(np.load(str(val_file)), batch_size=batch_size, shuffle=True)

    return train_dataiter, val_dataiter


class WithELU(HybridBlock):
    def __init__(self, block: HybridBlock, prefix: str = None):
        super().__init__(prefix=prefix)

        self._block = block

    def hybrid_forward(self, F, x, *args, **kwargs):
        y = self._block(x)
        return F.LeakyReLU(y, act_type='elu')


def _get_batch(data_iter: mx.io.DataIter):
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


class PlotGenerateImage:
    """
    A simple plot callback to plot an image sampled from the model.
    """
    def __init__(self, nn: HybridBlock, freq: int, image_shape: Union[Tuple[int, int], Tuple[int, int, int]],
                 conditioning_variables: Tuple[nd.NDArray] = tuple()):
        """
        :param nn: Generative model to sample from. Must implement generate method. See DRAW for an example.
        :param freq: Plotting frequency.
        :param image_shape: Image shape.
        """
        self._nn = nn
        self._freq = freq
        self._image_shape = image_shape
        self._conditioning_variables = conditioning_variables
        self._last_call = 0

    def __call__(self, mxb_writer: mxboard.SummaryWriter, samples_processed: int, *args, **kwargs):
        if samples_processed - self._last_call > self._freq:
            self._last_call = samples_processed
            # generate image from model
            img = self._nn.generate(*self._conditioning_variables).asnumpy()[0].reshape(self._image_shape)
            mxb_writer.add_image('Generated_image', img, samples_processed)


class PlotGradientHistogram:
    """
    A plot callback to plot histograms of gradient values.
    """
    def __init__(self, nn: HybridBlock, freq: int):
        """
        :param nn: Model to plot parameters of.
        :param freq: Plotting frequency.
        """
        self._params = nn.collect_params()
        self._freq = freq
        self._last_call = 0

    def __call__(self, mxb_writer: mxboard.SummaryWriter, samples_processed: int, *args, **kwargs):
        if samples_processed - self._last_call > self._freq:
            self._last_call = samples_processed
            for k, p in self._params.items():
                if p.grad_req != 'null':
                    g = p.grad().asnumpy()
                    mxb_writer.add_histogram(k, g, samples_processed, bins=10)


def train(forward_fn: Callable[[mx.io.DataBatch], nd.NDArray], train_iter: mx.io.DataIter, val_iter: mx.io.DataIter,
          trainer: Trainer, num_train_samples: int, num_val_samples: int, val_freq: int,
          logdir: str, run_suffix: str = '',
          plot_callbacks: Tuple[Callable[[mxboard.SummaryWriter, int], None]] = tuple()):
    """
    Train the model given by its forward function.

    :param forward_fn: The forward function of the model that takes in a batch and returns loss (over samples in a
        batch)
    :param train_iter: Training data iterator.
    :param val_iter: Validation data iterator.
    :param trainer: Trainer.
    :param num_train_samples: Number of training samples.
    :param num_val_samples: Number of validation samples (per validation).
    :param val_freq: Validation frequency (in number of samples).
    :param logdir: Log directory for mxboard.
    :param run_suffix: Suffix for run id.
    :param plot_callbacks: A list of additional plotting callbacks. These are called after each update. A plotting
      callback should expect a mxboard.SummaryWriter and the iteration number. See DRAW/train_mnist.py for an example.
    """
    run_id = '{}{}'.format(strftime('%Y%m%d%H%M%S'), run_suffix)
    sw = mxboard.SummaryWriter(logdir=os.path.join(logdir, run_id))
    pm = tqdm.tqdm(total=num_train_samples)

    last_val_loss = np.inf
    last_val_time = 0
    samples_processed = 0
    while samples_processed < num_train_samples:
        batch = _get_batch(train_iter)

        # train step
        with autograd.record():
            loss = forward_fn(batch)
        autograd.backward(loss)

        batch_size = loss.shape[0]
        trainer.step(batch_size=batch_size)

        samples_processed += batch_size
        last_train_loss = nd.mean(loss).asscalar()  # loss per sample
        # plot loss
        sw.add_scalar('Loss', {'Training': last_train_loss}, samples_processed)
        pm.update(n=batch_size)

        # call plot callbacks
        for callback in plot_callbacks:
            callback(sw, samples_processed)

        # validation step
        if samples_processed - last_val_time >= val_freq:
            last_val_time = samples_processed
            tot_val_loss = 0.0
            j = 0
            while j < num_val_samples:
                batch = _get_batch(val_iter)
                loss = forward_fn(batch)
                tot_val_loss += nd.sum(loss).asscalar()
                j += loss.shape[0]

            last_val_loss = tot_val_loss / j  # loss per sample
            sw.add_scalar('Loss', {'Validation': last_val_loss}, samples_processed)
            sw.flush()

        pm.set_postfix({'Train loss': last_train_loss, 'Val loss': last_val_loss})

    # calculate loss on the whole validation set
    tot_val_loss = 0.0
    j = 0
    for batch in val_iter:
        loss = forward_fn(batch)
        tot_val_loss += nd.sum(loss).asscalar()
        j += loss.shape[0]

    last_val_loss = tot_val_loss / j  # loss per sample
    sw.add_scalar('Loss', {'Validation_final': last_val_loss}, samples_processed)
    pm.set_postfix({'Train loss': last_train_loss, 'Val loss': last_val_loss})

    pm.close()
    sw.flush()
    sw.close()

    return run_id


def check_gradient(forward_fn, fn_params: List[mx.ndarray.NDArray], wrt: Parameter, seed=None, eps=3e-4,
                   tol=1e-2) -> bool:
    """
    Check autograd backward for a given function using finite differencing.

    :param forward_fn: The function to test the gradients of. This function should return a scalar.
    :param fn_params: A list of parameters to call the function.
    :param wrt: The parameter with respect to which we take the gradient.
    :param seed: Random seed for mxnet and numpy. Note that the forward function might be stochastic. We reinitialize
        the seed to the same number before every forward function call.
    :param eps: Epsilon used in finite differencing. The default value is taken from theano's verify_grad function.
    :param tol: Absolute and relative tolerance used to check equality. Again, the default value is taken from theano's
        verify_grad function.
    :return: True if check succeeds.
    """
    if seed is None:
        seed = int(np.random.rand() * 1e6)

    # calculate gradient with autograd
    mx.random.seed(seed)
    np.random.seed(seed)
    with autograd.record():
        out = forward_fn(*fn_params)

    autograd.backward(out)
    ag_grad = wrt.grad().asnumpy()

    # calculate gradient by finite difference
    orig_data = wrt.data().asnumpy()
    fd_grad = np.zeros_like(orig_data)
    for i in range(orig_data.size):
        ix = np.unravel_index(i, orig_data.shape)

        # f(x + h)
        orig_data[ix] += eps
        wrt.set_data(orig_data)
        mx.random.seed(seed)
        np.random.seed(seed)
        out_ph = forward_fn(*fn_params).asscalar()

        # f(x - h)
        orig_data[ix] -= (2*eps)
        wrt.set_data(orig_data)
        mx.random.seed(seed)
        np.random.seed(seed)
        out_mh = forward_fn(*fn_params).asscalar()
        orig_data[ix] += eps  # revert

        # calc gradient
        fd_grad[ix] = (out_ph - out_mh) / (2 * eps)

    return np.allclose(ag_grad, fd_grad, atol=tol, rtol=tol)


