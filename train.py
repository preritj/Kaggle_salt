import os
import functools
import tensorflow as tf
from data_reader import Dataset
from nn.unet import UNet


class Trainer(object):

    def __init__(self, cfg):
        # Define model parameters
        self.cfg = cfg
        self.hparams = tf.contrib.training.HParams(
            **self.cfg.__dict__)

    def get_features_labels_data(self):
        """returns dataset containing (features, labels)"""
        data_reader = Dataset(self.cfg)
        dataset = data_reader.read_data()

        def map_fn(image, mask, height):
            features = {'image': image,
                        'height': height}
            labels = {'mask': mask}
            return features, labels

        dataset = dataset.map(
            map_fn, num_parallel_calls=self.cfg.num_parallel_map_calls)
        dataset = dataset.prefetch(self.cfg.prefetch_size)
        dataset = dataset.repeat(self.cfg.num_epochs or None)
        dataset = dataset.batch(self.cfg.batch_size)
        dataset = dataset.prefetch(self.cfg.prefetch_size)
        return dataset

    def prepare_tf_summary(self, features, predictions, max_display=4):
        images = tf.cast(features['image'], tf.uint8)
        masks = tf.cast(predictions['mask'], tf.uint8)
        tf.summary.image('images', images, max_display)
        tf.summary.image('masks', masks, max_display)

    def train(self):
        """run training experiment"""
        session_config = tf.ConfigProto(
            allow_soft_placement=True
        )

        if not os.path.exists(self.cfg.model_dir):
            os.makedirs(self.cfg.model_dir)

        model_path = os.path.join(
            self.cfg.model_dir,
            self.cfg.model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.hparams.model_dir = model_path

        model_dir = model_path

        run_config = tf.contrib.learn.RunConfig(
            model_dir=model_dir,
            session_config=session_config
        )

        estimator = tf.estimator.Estimator(
            model_fn=self.get_model_fn(),
            params=self.hparams,  # HParams
            config=run_config  # RunConfig
        )

        hooks = None

        def train_input_fn():
            """Create input graph for model.
            """
            dataset = self.get_features_labels_data()
            return dataset

        # train_input_fn = self.input_fn
        estimator.train(input_fn=train_input_fn,
                        hooks=hooks)

    def get_optimizer_fn(self):
        """returns an optimizer function
        which takes as argument learning rate"""
        opt = dict(self.cfg.optimizer)
        opt_name = opt.pop('name', None)

        if opt_name == 'adam':
            opt_params = opt.pop('params', {})
            # remove learning rate if present
            opt_params.pop('learning_rate', None)

            def optimizer_fn(lr):
                opt = tf.train.AdamOptimizer(lr)
                return opt

        else:
            raise NotImplementedError(
                "Optimizer {} not yet implemented".format(opt_name))

        return optimizer_fn

    def get_train_op(self, loss):
        """Get the training Op.
        Args:
             loss (Tensor): Scalar Tensor that represents the loss function.
        Returns:
            Training Op
        """
        # TODO: build configurable optimizer
        # optimizer_cfg = train_cfg.optimizer

        learning_rate = self.cfg.learning_rate
        lr_decay_params = self.cfg.learning_rate_decay
        if lr_decay_params is not None:
            lr_decay_fn = functools.partial(
                tf.train.exponential_decay,
                decay_steps=lr_decay_params['decay_steps'],
                decay_rate=lr_decay_params['decay_rate'],
                staircase=True
            )
        else:
            lr_decay_fn = None

        return tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            optimizer=self.get_optimizer_fn(),
            learning_rate=learning_rate,
            learning_rate_decay_fn=lr_decay_fn
        )

    @staticmethod
    def get_eval_metric_ops(labels, predictions):
        """Return a dict of the evaluation Ops.
        Args:
            labels (Tensor): Labels tensor for training and evaluation.
            predictions (Tensor): Predictions Tensor.
        Returns:
            Dict of metric results keyed by name.
        """
        return {
            'Accuracy': tf.metrics.accuracy(
                labels=labels,
                predictions=predictions,
                name='accuracy')
        }

    def get_model_fn(self):
        """Return the model_fn.
        """

        def model_fn(features, labels, mode, params):
            """Model function used in the estimator.
            Args:
                model (Model): an instance of class Model
                features (Tensor): Input features to the model.
                labels (Tensor): Labels tensor for training and evaluation.
                mode (ModeKeys): Specifies if training, evaluation or prediction.
                params (HParams): hyperparameters.
            Returns:
                (EstimatorSpec): Model to be run by Estimator.
            """
            model = None
            model_name = params.model_name
            print("Using model ", model_name)
            if model_name == 'mobilenet_pose':
                model = UNet(params)
            else:
                NotImplementedError("{} not implemented".format(model_name))

            is_training = mode == tf.estimator.ModeKeys.TRAIN
            # Define model's architecture
            # inputs = {'images': features}
            # predictions = model.predict(inputs, is_training=is_training)
            predictions = model.predict(features, is_training=is_training)
            self.prepare_tf_summary(features, predictions)
            # Loss, training and eval operations are not needed during inference.
            loss = None
            train_op = None
            eval_metric_ops = {}
            if mode != tf.estimator.ModeKeys.PREDICT:
                # labels = tf.image.resize_bilinear(
                #     labels, size=params.output_shape)
                # heatmaps = labels[:, :, :, :-1]
                # masks = tf.squeeze(labels[:, :, :, -1])
                # labels = heatmaps
                # ground_truth = {'heatmaps': heatmaps,
                #                 'masks': masks}
                ground_truth = labels
                losses = model.losses(predictions, ground_truth)
                for loss_name, loss_val in losses.items():
                    tf.summary.scalar('loss/' + loss_name, loss_val)
                # with tf.device(self.param_server_device):
                loss = losses['seg_loss']
                train_op = self.get_train_op(loss)
                eval_metric_ops = None  # get_eval_metric_ops(labels, predictions)
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op,
                eval_metric_ops=eval_metric_ops
            )

        return model_fn