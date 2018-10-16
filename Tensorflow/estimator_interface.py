import os
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

from typing import List

from abc import ABC, abstractmethod

class EstimaterInterface(tf.estimator.Estimator):

    '''
    Base class that extends TensorFlow's estimator class.
    Will serve as the basis for the classifier and 
    the autoencoder, but can also be used to create
    other estimators.
    
    Constructor args: 
        logging_steps - How often logging will be performed
        checkpoint_steps - How often checkpoints will be saved
        summary_steps - How often new summaries will be added
        model_dir - Directory where model data will be stored 
        model_fn - A function that takes features, labels and the estimator's mode and returns an EstimatorSpec object.
        Refer to https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#__init__ for details.
        
	'''

    ACTIVATION_FN_DICT = {
        'sigmoid':tf.nn.sigmoid,
        'relu':tf.nn.relu
    }

    OPTIMIZER_DICT = {
        'adam': tf.train.AdamOptimizer,
        'adagrad': tf.train.AdagradOptimizer
    }

    def __init__(
        self, *, 
        activation_fns: List[str], neurons_per_layer: List[int], 
        optimizer: str, model_dir: str, logging_steps: int, 
        checkpoint_steps: int, summary_steps: int,
        train_input_fn=None, eval_input_fn=None):

        if not isinstance(neurons_per_layer, list):
            raise ValueError('"neurons_per_layer" must be a list of integers. You provided: {}'.format(type(neurons_per_layer)))

        if not isinstance(activation_fns, list):
            raise ValueError('"activation_fns" must be a list of integers. You provided: {}'.format(type(activation_fns)))

        for act_fn in activation_fns:
            if act_fn not in self.ACTIVATION_FN_DICT:
                raise ValueError('Invalid activation function provided: {}'.format(act_fn))

        if (len(activation_fns) != len(neurons_per_layer)):
            raise ValueError('The number of activation functions ({}) does not match the number of layers. ({})'.format(
                len(activation_fns), len(neurons_per_layer)
            ))

        if optimizer not in self.OPTIMIZER_DICT:
            raise ValueError('Invalid optimizer provided: {}'.format(optimizer))


        self.train_input_fn = train_input_fn
        self.eval_input_fn = eval_input_fn

        self.neurons_per_layer = neurons_per_layer

        self.activation_fn_names = activation_fns
        self.activation_fns = [self.ACTIVATION_FN_DICT[self.activation_fn_names[i]] for i in range(len(activation_fns))]

        self.optimizer_name = optimizer
        self.optimizer = self.OPTIMIZER_DICT[optimizer]

        self.ei_model_dir = model_dir
        self.ei_config=tf.estimator.RunConfig(
            log_step_count_steps=logging_steps,
            save_checkpoints_steps=checkpoint_steps,
            save_checkpoints_secs=None,
            save_summary_steps=summary_steps
        )

    def _compile(self, model_fn):
        super().__init__(
            model_fn=model_fn,
            model_dir=self.ei_model_dir,
            config=self.ei_config
        )

    def get_optimizer_list(self, optimizer_names: List[str]):
        return [self.OPTIMIZER_DICT[optimizer_names[i]] for i in range(len(optimizer_names))]

    def get_activation_fn_list(self, activation_fn_names:List[str]) -> List[tf.train.Optimizer]:
        return [self.ACTIVATION_FN_DICT[activation_fn_names[i]] for i in range(len(activation_fn_names))]

    def train_and_eval(self):
        pass
        # train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=1000)
        # eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

        # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    def train(self, train_features, batch_size, epochs, train_labels=None):
        '''
            Calls the train() method of tf.estimator.Estimator with input function, constructed from the provided parameters.

            Args:
                train_features - Pandas DataFrame object that contains all training features (no labels)
                batch_size - The batch size that will be passed to tf.Dataset.batch()
                epochs - The value that will be passed to tf.Dataset.repeat()
                train_labels - Pandas DataFrame object that contains labels only. Optional since not required in unsupervised learning tasks.
        '''

        input_fn = lambda:self._input_fn(train_features, batch_size, labels=train_labels, num_epochs=epochs, drop_last=True)

        if self.train_input_fn:
            input_fn = self.train_input_fn

        super().train(input_fn)

    def evaluate(self, test_features, test_labels=None):
        '''
            Calls the evaluate() method of tf.estimator.Estimator with input function, constructed from the provided parameters. Returns the observed metrics as defined in model_fn of the estimator through which this function is called.
            
            Args:
                test_features - Pandas DataFrame object that contains all training features (no labels)
                test_labels - Pandas DataFrame object that contains labels only. Optional since not required in unsupervised learning tasks.
        '''

        input_fn = lambda:self._input_fn(features=test_features, labels=test_labels, batch_size=len(test_features)//4, shuffle=False)

        if self.eval_input_fn:
            input_fn = self.eval_input_fn

        return super().evaluate(input_fn)


    def _input_fn(self, features, batch_size, labels=None, num_epochs=None, shuffle=True, drop_last=True):
        '''
            This function is passed to train() and evaluate(). It converts the features provided to the two functions to a tf.data.Dataset object. Batching and shuffling is optionally performed. The number of repetitions of the dataset until end of sequence error is thrown is given via num_epochs. Finally the dataset object is returned.
        '''
        if labels is not None: dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
        else: dataset = tf.data.Dataset.from_tensor_slices((dict(features)))

        if shuffle:
            dataset = dataset.shuffle(len(features))
        
        if num_epochs:
            dataset = dataset.repeat(num_epochs)

        # dataset = dataset.batch(batch_size)
        
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        dataset = dataset.prefetch(1)
        return dataset

    def get_weight_and_bias(self, layer_name):
        '''
        Args: 
            layer (Tensor) - The layer for which the weight and the bias are to be returned

        Returns: 
            A tuple of tensors in the form (weight,bias)
        '''
        
        weight_matrix, bias_vector = None, None
        try:
            weight_matrix = self.get_variable_value(layer_name + "/kernel")
            bias_vector = self.get_variable_value(layer_name + "/bias")
        except ValueError:
            raise ValueError('Can\'t get weights and biases for untrained estimator.')
        
        return weight_matrix, bias_vector

        # with tf.variable_scope(os.path.split(layer.name)[0], reuse=True):
        #     return tf.get_variable("kernel"), tf.get_variable("bias") 
    
    @abstractmethod
    def __build_prediction(self):
        '''Converts the output into a prediction that can be compared to a label.'''
        pass

    @abstractmethod
    def __loss(self):
        '''Defines the loss function that will be optimized'''
        pass

    @abstractmethod
    def __build_network(self):
        '''Defines the structure of the network'''
        pass

    @abstractmethod
    def __build_estimator(self):
        '''Defines the process of training, prediction and evaluation'''
        pass