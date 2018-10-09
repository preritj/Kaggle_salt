import os
import pandas as pd
import tensorflow as tf
from glob import glob
import functools
import tfrecord_util

slim_example_decoder = tf.contrib.slim.tfexample_decoder


class Dataset(object):
    def __init__(self, cfg):
        self.cfg = cfg
        if not os.path.exists(cfg.tfrecord_dir):
            self.create_tfrecords()

    def _create_tf_example(self, img_file, mask_file, depth):
        feature_dict = {
            'image/filename':
                tfrecord_util.bytes_feature(img_file.encode('utf8')),
            'mask/filename':
                tfrecord_util.bytes_feature(mask_file.encode('utf8')),
            'depth':
                tfrecord_util.float_feature(depth)
        }
        return tf.train.Example(features=tf.train.Features(feature=feature_dict))

    def create_tfrecords(self):
        cfg = self.cfg
        print("Creating tf records")
        os.makedirs(cfg.tfrecord_dir)
        mask_files = glob(os.path.join(cfg.mask_dir, '*.png'))
        img_files = [os.path.join(cfg.img_dir, os.path.basename(f)) for f in mask_files]
        data = pd.read_csv(cfg.depth_file, index_col=0)
        trains_ids = [os.path.basename(f).split(".png")[0] for f in mask_files]
        z_values = data.loc[trains_ids]['z']
        filename = os.path.join(cfg.tfrecord_dir, "0.tfrecord")
        writer = tf.python_io.TFRecordWriter(filename)
        for img_file, mask_file, depth in zip(img_files, mask_files, z_values):
            tf_example = self._create_tf_example(img_file, mask_file, depth)
            writer.write(tf_example.SerializeToString())

    @staticmethod
    def _image_decoder(keys_to_tensors):
        filename = keys_to_tensors['image/filename']
        image_string = tf.read_file(filename)
        # TODO: decode after crop to increase speed
        image_decoded = tf.image.decode_png(image_string, channels=3)
        return image_decoded

    @staticmethod
    def _mask_decoder(keys_to_tensors):
        filename = keys_to_tensors['mask/filename']
        image_string = tf.read_file(filename)
        # TODO: decode after crop to increase speed
        mask_decoded = tf.image.decode_png(image_string, channels=3)
        return mask_decoded

    @staticmethod
    def _depth_decoder(keys_to_tensors):
        depth = keys_to_tensors['depth']
        return depth / 1000.

    def _decoder(self):
        keys_to_features = {
            'image/filename':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'mask/filename':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'depth':
                tf.FixedLenFeature((), tf.float32)
        }
        items_to_handlers = {
            'image': slim_example_decoder.ItemHandlerCallback(
                'image/filename', self._image_decoder),
            'mask': slim_example_decoder.ItemHandlerCallback(
                'mask/filename', self._mask_decoder),
            'depth': slim_example_decoder.ItemHandlerCallback(
                'depth', self._depth_decoder)
        }
        decoder = slim_example_decoder.TFExampleDecoder(keys_to_features,
                                                        items_to_handlers)
        return decoder

    def read_data(self):
        cfg = self.cfg
        decoder = self._decoder()
        tfrecord_files = glob(os.path.join(cfg.tfrecord_dir, "*.tfrecord"))
        dataset = tf.data.TFRecordDataset(tfrecord_files)

        if cfg.shuffle:
            dataset = dataset.shuffle(cfg.shuffle_buffer_size)

        decode_fn = functools.partial(
            decoder.decode, items=['image', 'mask', 'depth'])
        dataset = dataset.map(
            decode_fn, num_parallel_calls=cfg.num_parallel_map_calls)
        dataset = dataset.prefetch(cfg.prefetch_size)

        