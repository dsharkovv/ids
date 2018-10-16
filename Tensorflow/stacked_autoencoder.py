import os 
import tensorflow as tf
from typing import List
import util
from Tensorflow.sparse_autoencoder import SparseAutoencoder

class StackedAutoencoder():
    
    ACTIVATION_FN_DICT = {
        'sigmoid':tf.nn.sigmoid,
        'relu':tf.nn.relu
    }

    def __init__(
        self, *, feature_columns, hidden_layers: List[int], 
        weight_decay: List[float], rho: List[float], 
        kl_weight: List[float], logging_steps=1000,
        activation_fns:List[str], dropout=None, batch_norm=None, 
        checkpoint_steps=100000, summary_steps=1000, logger=None,
        model_dir='stacked_autoencoder'):

        '''
          Args: 
            feature_columns - The feature columns that are used to build the dataset
            hidden_layers - A a list of integers, specifying how many hidden units each autoencoder will have
            weight_decay - A list of floats that sets the strength of the weight decay in each autoencoder
            rho - A list of floats specifying the desired average activation in each autoencoder
            kl_weight - A list of floats specfiying the strength of the KL-divergence in each autoencoder
            logging_steps - How often metrics should be logged
            checkpoint_steps - How often checkpoints should be saved
            summary_steps - How often summaries should be written
            model_dir - The directory where data will be stored
        '''

        if not isinstance(hidden_layers, list):
            raise ValueError('"hidden_layers" must be a list. You provided: {}'.format(type(hidden_layers)))
            
        if not isinstance(activation_fns, list):
            raise ValueError('"activation_fns" must be a list. You provided: {}'.format(type(activation_fns)))

        if not isinstance(weight_decay, list):
            raise ValueError('"weight_decay" must be a list. You provided: {}'.format(type(weight_decay)))

        if not isinstance(rho, list):
            raise ValueError('"rho" must be a list. You provided: {}'.format(type(rho)))

        if not isinstance(kl_weight, list):
            raise ValueError('"kl_weight" must be a list. You provided: {}'.format(type(kl_weight)))

        if len(hidden_layers) != len(weight_decay) or \
           len(weight_decay) != len(kl_weight):
            raise ValueError('All lists must have the same length.')

        self.logger = logger

        self.activation_fn_names = activation_fns
        self.activation_fns = [self.ACTIVATION_FN_DICT[self.activation_fn_names[i]] for i in range(len(activation_fns))]
        
        self.dropout=False if dropout is None else dropout
        self.batch_norm = False if batch_norm is None else batch_norm

        self.hidden_layers = hidden_layers
        self.weight_decay = weight_decay
        self.kl_weight = kl_weight
        self.rho = rho
        self.feature_columns = feature_columns

        self.checkpoint_steps = checkpoint_steps
        self.logging_steps = logging_steps
        self.summary_steps = summary_steps

        self.model_dir = model_dir
        self.autoencoder_dirs = [None for _ in range(len(hidden_layers))]
        self.autoencoders = []

    @property
    def neurons_per_layer(self):
        return self.hidden_layers

    def __str__(self):

        msg = '({})-->'
        
        if self.autoencoders and self.autoencoders[0].network_built:
            params = [self.autoencoders[0].input_layer.get_shape().as_list()[1]]
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
        msg+='({}) '.format(params[0])
        return msg.format(*params)

    def train(self, *, train_ds, batch_size:int, epochs:List[int], learning_rates:List[int], test_ds=None):
        '''
          Args: 
            train_ds - The training dataset
            batch_size - The batch size to be used with gradient descent
            epochs - A a list of integers, specifying for how many epochs each autoencoder will be trained
            learning_rates - A list of floats that sets the learning rate in each autoencoder
            batch_norm - Boolean value indicating whether to apply batch normalization to the hidden layers
            dropout - Boolean value indicating whether to apply dropout to the hidden layers
            test_ds - Optional dataset on which to perform evaluation
        '''
        if not isinstance(epochs, list):
            raise ValueError('"epochs" must be a list. You provided: {}'.format(type(epochs)))

        if not isinstance(learning_rates, list):
            raise ValueError('"epochs" must be a list. You provided: {}'.format(type(learning_rates)))

        if len(epochs) != len(self.hidden_layers) or \
           len(epochs) != len(learning_rates):
            raise ValueError('All list parameters must have the same length')

        weights, biases = [], []
        hidden_units, act_fns = [], []

        for i in range(len(self.hidden_layers)):
            transformation_fn = None
            if i > 0: transformation_fn = self.build_transformation_fn(weights, biases, act_fns, hidden_units)

            if self.autoencoder_dirs[i] is None:
                self.autoencoder_dirs[i] = os.path.join(self.model_dir, 'ae_{}'.format(i))

            model_dir = self.autoencoder_dirs[i]
            util.makedir_maybe(model_dir)

            ae = SparseAutoencoder(
                feature_columns=self.feature_columns, kl_weight=self.kl_weight[i], 
                weight_decay=self.weight_decay[i], num_hidden_neurons=self.hidden_layers[i],
                learning_rate=learning_rates[i], transformation_fn=transformation_fn, 
                logging_steps=self.logging_steps, checkpoint_steps=self.checkpoint_steps, 
                summary_steps=self.summary_steps, dropout=self.dropout, rho=self.rho[i],
                batch_norm=self.batch_norm, model_dir=model_dir,activation_fn=self.activation_fn_names[i]
            )

            for _ in range(epochs[i]):
                ae.train(train_ds, batch_size, 1)
                if test_ds is not None: ae.evaluate(test_ds)

            self.autoencoders.append(ae)

            weight, bias = ae.get_weights_and_biases()
            num_hidden = ae.n_hidden
            act_fn = ae.activation_fn
            weights.append(weight)
            biases.append(bias)
            hidden_units.append(num_hidden)
            act_fns.append(act_fn)

    def get_weights_and_biases(self): 
        weights, biases = [], []
        for ae in self.autoencoders:
            w, b = ae.get_weights_and_biases()
            weights.append(w)
            biases.append(b)

        return weights, biases

    def evaluate(self, ae_idx, test_ds):
        if not self.autoencoders: 
            raise ValueError('Evaluation can only be performed once training has run.')
        
        self.autoencoders[ae_idx].evaluate(test_ds)

    def build_transformation_fn(self, weights, biases, activation_fns, hidden_layer_units):
        if not weights or not biases: return None

        def transformation_fn(x):
            with tf.name_scope('transformation_fn'):
                for i, (weight, bias, activation_fn, hlu) in enumerate(zip(weights, biases, activation_fns, hidden_layer_units)):

                    kernel_initializer = tf.constant_initializer(value=weight, verify_shape=True) 
                    bias_initializer = tf.constant_initializer(value=bias, verify_shape=True) 

                    layer = tf.layers.dense(
                        x, 
                        units=hlu, name='transf_fn_layer_{}'.format(i), 
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,trainable=False
                    )
                        
                    layer = activation_fn(layer)
                    x = layer

                return x

        return transformation_fn
