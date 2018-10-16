import os
import tensorflow as tf
from typing import List, Any

from Tensorflow.estimator_interface import EstimaterInterface

class Classifier(EstimaterInterface):

    def __init__(
        self, *, feature_columns, hidden_layers:List[int], 
        num_classes:int, non_trainable:List[int]=None,
        weight_decay = 0.001, weights:List[Any] = None, biases:List[Any] = None,
        learning_rate=0.001, activation_fns=None, optimizer='adam',
        train_last_only=None, dropout=None, batch_norm=None, model_dir="classifier", 
        logger=None, logging_steps=1000, checkpoint_steps=100000, summary_steps=1000):

        '''
          Args: 
            feature_columns - The feature columns that are used to build the dataset
            hidden_layers - A list with the number of hidden units in each layer. The length of the list
            is the total number of hidden layers. An empty lists will result in a softmax classifier.
            num_classes - Number of classes in the dataset. Determines the number of units in the output layer.
            non_trainable - A list of indices of the layers for which to disable training
            weight_decay - Decides the strength of the weight decay, applied to the loss function. 
            weights - A list of weights to be used for initialization. Must have the same length as hidden_layers
            biases - A list of biases to be used for initialization. Must have the same length as hidden_layers
            learning_rate - Learning rate used by the optimizer
            activation_fns - A list of the names of the activation functions to be applied. Must have the same length as hidden_layers
            optimizer - The name of the optimizer to be used
            dropout - Boolean value indicating whether to apply dropout to the hidden layers
            batch_norm - Boolean value indicating whether to apply batch normalization to the hidden layers
            train_last_only - Disables training on all layers, but the last
            model_dir - The directory where data will be stored
            logging_steps - How often metrics should be logged
            checkpoint_steps - How often checkpoints should be saved
            summary_steps - How often summaries should be written
        '''

        super().__init__(
            neurons_per_layer=hidden_layers, activation_fns=activation_fns,
            logging_steps=logging_steps, checkpoint_steps=checkpoint_steps,
            summary_steps=summary_steps, optimizer=optimizer, 
            model_dir=model_dir
        )

        if(weights is not None and biases is None) or \
          (weights is None and biases is not None):
            raise ValueError('Either no weights and no biases or both weights and biases have to be specified.')

        if weights is not None and biases is not None and \
           (len(hidden_layers) != len(weights) or \
           len(weights) != len(biases)):
            raise ValueError('Provided number of weights or biases does not match the number of layers.') 

        if not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError('"num_classes" must be a non-zero positive integer. You provided: {}'.format(num_classes)) 

        self.logger = logger

        self.params_provided = weights is not None and biases is not None
        self.dropout = False if dropout is None else dropout
        self.batch_norm = False if batch_norm is None else batch_norm
        self.train_last_only = False if train_last_only is None else train_last_only

        # Pre-initialized parameters
        self.weights = weights
        self.biases = biases

        self.learning_rate = learning_rate

        self.num_classes = num_classes
        self.weight_decay = weight_decay

        if non_trainable:
            self.trainable_layers = [i not in non_trainable for i in range(len(hidden_layers))]
        else: 
            if self.train_last_only: 
                self.trainable_layers = [False for _ in range(len(hidden_layers))]
                self.trainable_layers[-1] = True

            self.trainable_layers = [True for _ in range(len(hidden_layers))]

        self.input_layer = None
        self.hidden_layers = []
        self.output_layer = None
        self.labels = None
        self.weight_decay_cur = None
        self.weights_loaded = False

        self.network_built = False
        
        def _model_fn(features, labels, mode):
            self.labels = labels
            self.__build_network(features=features, feature_columns=feature_columns, mode=mode)
            self.network_built = True
            return self.__build_estimator(mode=mode)
            
        self._compile(_model_fn)

    def __str__(self):
        msg = '({})-->'
        if self.network_built:
            params = [self.input_layer.get_shape().as_list()[1]]
        else: params = ['?']

        for i, neurons in enumerate(self.neurons_per_layer):
            msg += '[({})->({})'

            params.extend([neurons, self.activation_fn_names[i]])

            if self.dropout: 
                msg += '->({})'
                params.append('do')

            if self.batch_norm:
                msg += '->({})'
                params.append('bn')

            msg += ']-->'
        msg+='({}) '.format(self.num_classes)
        # msg+='Trainable: {}'.format(self.get_trainable_params())
        return msg.format(*params)

    def __loss(self, output_layer, labels):
        if self.weight_decay > 0:
            weight_decay_term = tf.add_n(
                [tf.reduce_sum(var) for var in tf.losses.get_regularization_losses()]
            )
            self.weight_decay_cur = tf.identity(weight_decay_term, name="result")
            return tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=output_layer) + 0.5 * weight_decay_term)
        return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=output_layer) 

    def __build_prediction(self, logits):
        return tf.argmax(logits, 1)

    def __build_network(self, features, feature_columns, mode):
        if self.trainable_layers and self.trainable_layers[0]: 
            self.input_layer = tf.feature_column.input_layer(features, feature_columns)
        else:
            self.input_layer = tf.feature_column.input_layer(features, feature_columns, trainable=False)

        cur_input = self.input_layer
        l2_regularizer = tf.contrib.layers.l2_regularizer(scale=self.weight_decay)
        for i, n_neurons in enumerate(self.neurons_per_layer):

            ker_init, bias_init = None, None
            if self.params_provided and not self.weights_loaded: 
                if self.logger: self.logger.info('Loading weights for layer #{}'.format(i))
                weight, bias = self.weights[i], self.biases[i]
                ker_init = tf.constant_initializer(value=weight, verify_shape=True) 
                bias_init = tf.constant_initializer(value=bias, verify_shape=True) 

            self.hidden_layers.append( 
                tf.layers.dense(
                    cur_input, 
                    kernel_initializer=ker_init,
                    kernel_regularizer=l2_regularizer,
                    bias_initializer=bias_init,
                    units=n_neurons, 
                    trainable=self.trainable_layers[i],
                    activation=self.activation_fns[i],
                    name='hidden_layer_{}'.format(i)
                )
            )
            cur_input = self.hidden_layers[-1]
            if self.dropout: cur_input = tf.layers.dropout(cur_input, rate=0.5)
            if self.batch_norm: cur_input = tf.layers.batch_normalization(cur_input)

        self.output_layer = tf.layers.dense(
            cur_input, units=self.num_classes, 
            kernel_regularizer=l2_regularizer, 
            activation=None, name='output_layer'
        )

        self.weights_loaded = True

    def __build_estimator(self, mode):
        with tf.name_scope("loss"):
            loss = self.__loss(self.output_layer, self.labels)

        with tf.name_scope("prediction"):
            predictions = self.__build_prediction(self.output_layer)

        with tf.name_scope("metrics"):
            labels_one_dim = tf.squeeze(self.labels)
            update_ops = []
            accuracy, update_op_acc = tf.metrics.accuracy(
                labels=labels_one_dim,
                predictions=predictions,
                name='acc_op'
            )   
            update_ops.append(update_op_acc)
            
            metrics = {'accuracy': (accuracy, update_op_acc)}
            
            if self.num_classes == 2:
                labels_as_bool = tf.map_fn(lambda x: tf.equal(x, tf.zeros(x.shape, dtype=tf.int64)), labels_one_dim, dtype=tf.bool)
                predictions_as_bool = tf.map_fn(lambda x: tf.equal(x, tf.zeros(x.shape, dtype=tf.int64)), predictions, dtype=tf.bool)
                recall, update_op_rec = tf.metrics.recall(labels=labels_as_bool, predictions=predictions_as_bool)
                precision, update_op_prec = tf.metrics.precision(labels=labels_as_bool, predictions=predictions_as_bool)
                update_ops.append(update_op_rec)
                update_ops.append(update_op_prec)
                metrics['recall'] = (recall, update_op_rec)
                metrics['precision'] = (precision, update_op_prec)

            with tf.control_dependencies([loss]):
                tf.group(update_ops)
                tf.summary.scalar('accuracy', update_op_acc)
                if self.num_classes == 2:
                    tf.summary.scalar('recall', update_op_rec)
                    tf.summary.scalar('precision', update_op_prec)
                    tf.summary.scalar('f-measure', (2*precision*recall)/(recall+precision)) #TODO: FIX
                    if self.weight_decay > 0:
                        tf.summary.scalar("wd", self.weight_decay_cur)

        if mode == tf.estimator.ModeKeys.TRAIN:
            with tf.name_scope('train'):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

            # Add histograms for trainable variables
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)

            estimator_spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        elif mode == tf.estimator.ModeKeys.EVAL:
            estimator_spec = tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        else: 
            predictions_spec = {
                'class_ids': predictions[:, tf.newaxis],
                'probabilities': tf.nn.softmax(self.output_layer),
                'logits': self.output_layer,
            }
            estimator_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions_spec)

        return estimator_spec