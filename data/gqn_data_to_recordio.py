#!/usr/bin/python3
import argparse
import collections
import pickle
from pathlib import Path

import mxnet as mx
import tensorflow as tf
import tqdm


"""
This script creates a MXNetIndexedRecordIO file from a GQN dataset. This record file is then used by a custom data
iterator to read minibatches of scenes for GQN training.

Download the GQN dataset you are interested in from https://github.com/deepmind/gqn-datasets and put into this folder.
For example, if you want the shepard_metzler_5_parts dataset, copy that folder from the GCP bucket to this folder so
you have folders
  - data/shepard_metzler_5_parts/train
  - data/shepard_metzler_5_parts/test

Run this script to create record IO files for training and test sets.
common/get_gqn_data function needs the record IO files to instantiate the MXNet data iterators used for
training/evaluation.

NOTE that tensorflow JPG decoding and MXNet JPG decoding give slightly different results, so the images in the MXNet
record IO file will not be exactly the same with the ones in tfrecord files.
"""

# The following are taken from https://github.com/deepmind/gqn-datasets/blob/master/data_reader.py.
DatasetInfo = collections.namedtuple(
    'DatasetInfo',
    ['basepath', 'train_size', 'test_size', 'frame_size', 'sequence_size']
)


_DATASETS = dict(
    jaco=DatasetInfo(
        basepath='jaco',
        train_size=3600,
        test_size=400,
        frame_size=64,
        sequence_size=11),

    mazes=DatasetInfo(
        basepath='mazes',
        train_size=1080,
        test_size=120,
        frame_size=84,
        sequence_size=300),

    rooms_free_camera_with_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_with_object_rotations',
        train_size=2034,
        test_size=226,
        frame_size=128,
        sequence_size=10),

    rooms_ring_camera=DatasetInfo(
        basepath='rooms_ring_camera',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    rooms_free_camera_no_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_no_object_rotations',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    shepard_metzler_5_parts=DatasetInfo(
        basepath='shepard_metzler_5_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15),

    shepard_metzler_7_parts=DatasetInfo(
        basepath='shepard_metzler_7_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15)
)
_NUM_RAW_CAMERA_PARAMS = 5


def convert_dataset(dataset_name, dataset_type):
    # See https://github.com/deepmind/gqn-datasets/blob/master/data_reader.py for how to read the tfrecord files.

    feature_map = {
            'frames': tf.FixedLenFeature(
                shape=15, dtype=tf.string),
            'cameras': tf.FixedLenFeature(
                shape=[15 * 5],
                dtype=tf.float32)
    }

    def parse_fn(proto):
        return tf.parse_single_example(proto, feature_map)

    # read tfrecord files into a dataset
    files = list(map(lambda p: str(p), Path(dataset_name, dataset_type).glob('*.tfrecord')))
    ds = tf.data.TFRecordDataset(files)
    ds_info = _DATASETS[dataset_name]
    pds = ds.map(parse_fn)  # parse records into examples

    # create mxnet recordio file
    recordio_file = mx.recordio.MXIndexedRecordIO('{}_{}.idx'.format(dataset_name, dataset_type),
                                                  '{}_{}.rec'.format(dataset_name, dataset_type),
                                                  flag='w', key_type=int)

    pb = tqdm.tqdm()
    for i, tf_rec in enumerate(pds):
        frames = [tf.image.decode_jpeg(frame).numpy() for frame in tf_rec['frames']]
        assert len(frames) == ds_info.sequence_size

        cameras = tf_rec['cameras'].numpy().reshape((ds_info.sequence_size, _NUM_RAW_CAMERA_PARAMS))

        views = [mx.recordio.pack_img(mx.recordio.IRHeader(flag=0, label=camera, id=i, id2=j),
                                      img=frame,
                                      img_fmt='.jpg',
                                      quality=95)
                 for j, (frame, camera) in enumerate(zip(frames, cameras))]

        recordio_file.write_idx(i, pickle.dumps(views))
        pb.update(1)
    pb.close()

    recordio_file.close()


def main():
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='shepard_metzler_5_parts')
    args = parser.parse_args()

    # convert train and test datasets
    convert_dataset(args.dataset, 'train')
    convert_dataset(args.dataset, 'test')


if __name__ == "__main__":
    main()
