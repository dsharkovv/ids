import os
import tensorflow as tf

from Tensorflow.estimator_interface import EstimaterInterface

class SparseAutoencoder(EstimaterInterface):

    def __init__(
        self, *, feature_columns, 
        num_hidden_neurons:int, transformation_fn=None,
        kl_weight=0.01, rho=0.01, weight_decay=0.01, 
        learning_rate=0.001, activation_fn='sigmoid', 
        optimizer='adam', model_dir='autoencoder', batch_norm=None,
        dropout=None, logging_steps=1000,
        checkpoint_steps=100000, summary_steps=1000):

        '''
          Args: 
            feature_columns - The feature columns that are used to build the dataset
            num_hidden_neurons - An integer specifying the number of units in the hidden layer
            transformation_fn - A functions that creates a mapping from the input layer. The mapping is then used as the new input.
            kl_weight - A float specfiying the strength of the KL-divergence term
            rho - A float specifying the desired average activation of the hidden layer
            weight_decay - Decides the strength of the weight decay, applied to the loss function. 
            learning_rate - Learning rate used by the optimizer
            activation_fn - The name of the activation function to be applied
            optimizer - The name of the optimizer to be used
            model_dir - The directory where data will be stored
            batch_norm - Boolean value indicating whether to apply batch normalization to the hidden layers
            dropout - Boolean value indicating whether to apply dropout to the hidden layers
            logging_steps - How often metrics should be logged
            checkpoint_steps - How often checkpoints should be saved
            summary_steps - How often summaries should be written
        '''

        super().__init__(
            neurons_per_layer=[num_hidden_neurons], activation_fns=[activation_fn],
            logging_steps=logging_steps, checkpoint_steps=checkpoint_steps,
            summary_steps=summary_steps, model_dir=model_dir, optimizer=optimizer
        )

        self.dropout = False if dropout is None else dropout
        self.batch_norm = False if batch_norm is None else batch_norm

        self.input_layer = None
        self.transformation_fn = transformation_fn
        self.kl_weight = kl_weight
        self.rho = rho
        self.weight_decay = weight_decay
        self.activation_fn_name = self.activation_fn_names[0]
        self.activation_fn = self.activation_fns[0]
        self.output_activation_fn = self.ACTIVATION_FN_DICT['sigmoid']
        self.n_hidden = self.neurons_per_layer[0]
        self.learning_rate = learning_rate

        self.kl_cur = 0
        self.weight_decay_cur = None

        self.network_built = False

        def _model_fn(features, mode):
            self.__build_network(features=features, feature_columns=feature_columns, mode=mode)
            self.network_built = True
            return self.__build_estimator(mode=mode)

        self._compile(_model_fn)

    # def train_and_eval(self): 
    #     train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=1000)
    #     eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

    #     tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    def __str__(self):
        if self.network_built:
            input_size = [self.input_layer.get_shape().as_list()[1]]
        else: input_size = ['?']
        
        msg = '({})-->[({})->({})'

        params = [
            input_size, 
            self.n_hidden, 
            self.activation_fn_name
        ]

        if self.dropout: 
            msg += '->({})'
            params.append('do')

        if self.batch_norm:
            msg += '->({})'
            params.append('bn')

        msg += ']-->({})'
        params.append(input_size)
        return msg.format(*params)

    def get_weights_and_biases(self):
        return super().get_weight_and_bias('hidden_layer')
        # weight_matrix, bias_vector = None, None
        # try:
        #     weight_matrix = self.get_variable_value("hidden_layer/kernel")
        #     bias_vector = self.get_variable_value("hidden_layer/bias")
        # except ValueError:
        #     raise ValueError('Can\'t get weights and biases for untrained estimator.')
        
        # return weight_matrix, bias_vector

    # Calculates the loss based on the formula provided in the paper
    # "A Deep Learning Approach for Network Intrusion Detection System"
    def __loss(self):
        with tf.name_scope('sum_of_squares'):
            sum_of_squares_term = tf.losses.mean_squared_error(labels=self.input_layer, predictions=self.output_layer)
            tf.identity(sum_of_squares_term, name="result")
 
        with tf.name_scope('weight_decay'):
            weight_decay_term = tf.add_n(
                [tf.reduce_sum(var) for var in tf.losses.get_regularization_losses()]
            )
            self.weight_decay_cur = tf.identity(weight_decay_term, name="result")

        with tf.name_scope('kl'):
            kl_term = self.__kl_divergence(self.rho, tf.reduce_mean(self.hidden_layer))
            self.kl_cur = tf.identity(kl_term, name="result")

        return 0.5 * sum_of_squares_term + 0.5 * weight_decay_term + self.kl_weight * kl_term

    def __kl_divergence(self, rho, rho_hat):
        return rho * tf.log(tf.div(rho, rho_hat)) + (1 - rho) * tf.log(tf.div(1 - rho, 1 - rho_hat))

    # Sets up the graph by turning features into a tensor via the
    # feature columns and building the layers. Returns the resulting network and 
    # the input tensor. The latter to be used  to compare the output with the input.
    def __build_network(self, features, feature_columns, mode):
        if self.transformation_fn:
            self.input_layer = tf.feature_column.input_layer(features, feature_columns, trainable=False)
            self.input_layer = self.transformation_fn(self.input_layer)
        else: self.input_layer = tf.feature_column.input_layer(features, feature_columns)

        l2_regularizer = tf.contrib.layers.l2_regularizer(scale=self.weight_decay)

        self.hidden_layer = tf.layers.dense(self.input_layer, units=self.n_hidden, kernel_regularizer=l2_regularizer, name='hidden_layer')
        self.hidden_layer = self.activation_fn(self.hidden_layer)

        cur_input = self.hidden_layer
        if self.dropout: cur_input = tf.layers.dropout(cur_input)
        if self.batch_norm: cur_input = tf.layers.batch_normalization(cur_input)

        self.output_layer = tf.layers.dense(
            cur_input, 
            units=self.input_layer.shape[1].value,  
            kernel_regularizer=l2_regularizer, 
            activation=self.output_activation_fn, 
            name='output_layer'
        )

    def __build_prediction(self, logits):
        protocol = tf.argmax(logits[:,:3], axis=1)
        service = tf.argmax(logits[:,3:69], axis=1)
        flag = tf.argmax(logits[:,69:80], axis=1) 

        one_hot_protocol = tf.one_hot(protocol,3)
        one_hot_service = tf.one_hot(service,66)
        one_hot_flag = tf.one_hot(flag,11)

        bool_vals = logits[:,80:85]
        bool_vals = tf.map_fn(lambda x: tf.map_fn(lambda y: tf.cond(y < 0.5, lambda: 0., lambda: 1.), x), bool_vals)
        
        return tf.concat([
            one_hot_protocol, one_hot_service, 
            one_hot_flag, bool_vals, 
            logits[:,85:]],1, name='result'
        )

    def __build_estimator(self, mode):

        # hidden = tf.identity(self.hidden_layer, name="hidden")
        # logits = tf.identity(self.input_layer, name="input")

        with tf.name_scope("loss"):
            loss = self.__loss()

        with tf.name_scope("metrics"):
            mse, update_op = tf.metrics.mean_squared_error(labels=self.input_layer, predictions=self.output_layer, name="mse")
            metrics = {'mse': (mse, update_op)}
            with tf.control_dependencies([loss]):
                tf.group(update_op)
                tf.summary.scalar('mse', update_op)
                tf.summary.scalar("kl", self.kl_cur)
                tf.summary.scalar("wd", self.weight_decay_cur)

        if mode == tf.estimator.ModeKeys.TRAIN:

            with tf.name_scope('train'):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss, global_step=tf.train.get_global_step())

            # Add histograms for trainable variables
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
            tf.summary.histogram("hidden_layer/activations", self.hidden_layer)
            tf.summary.histogram("output_layer/activations", self.output_layer)

            estimator_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op
            )

        elif mode == tf.estimator.ModeKeys.EVAL:
            estimator_spec = tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        else: 
            with tf.name_scope("prediction"):
                predictions = self.__build_prediction(self.output_layer)

            estimator_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        return estimator_spec