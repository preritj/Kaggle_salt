from abc import abstractmethod
import tensorflow as tf


EPSILON = 1e-5


class Model:
    def __init__(self, model_cfg):
        self.cfg = model_cfg
        self._num_keypoints = self.cfg.num_keypoints
        self._num_vecs = self.cfg.num_vecs
        self.check_output_shape()

    @abstractmethod
    def check_output_shape(self):
        """Check shape consistency"""
        raise NotImplementedError("Not yet implemented")

    @abstractmethod
    def preprocess(self, inputs):
        """Image preprocessing"""
        raise NotImplementedError("Not yet implemented")

    @abstractmethod
    def build_net(self, preprocessed_inputs, is_training=False):
        """Builds network and returns heatmaps and fpn features"""
        raise NotImplementedError("Not yet implemented")

    def predict(self, inputs, is_training=False):
        images = inputs['image']
        heights = inputs['height']
        preprocessed_images = self.preprocess(images)
        masks = self.build_net(
            preprocessed_images, heights, is_training=is_training)
        prediction = {'masks': masks}
        return prediction

    def seg_loss(self, labels, logits):
        return tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits)

    def losses(self, prediction, ground_truth):
        mask_logits = prediction['masks']
        mask_gt = ground_truth['mask']

        seg_loss = self.seg_loss(
            labels=mask_gt,
            logits=mask_logits)

        losses = {'seg_loss': seg_loss}
        # l2_loss = tf.losses.mean_squared_error(
        #     heatmaps_gt, heatmaps_pred,
        #     reduction=tf.losses.Reduction.NONE)
        # l2_loss = weights * tf.reduce_mean(l2_loss, axis=-1)
        # l2_loss = tf.reduce_mean(l2_loss)
        # # TODO : add regularization losses
        # losses = {'l2_loss': l2_loss}
        return losses
