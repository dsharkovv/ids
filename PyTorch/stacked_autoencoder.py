import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import util 
from typing import List

from tensorboardX import SummaryWriter

from PyTorch.abstract_estimator import AbstractEstimator
from PyTorch.autoencoder import SparseAutoencoder
from PyTorch.callback import EstimatorCallback

class StackedAutoencoder(AbstractEstimator):
    
    class SAECallback(EstimatorCallback):
        '''Callback to swtich from one autoencoder to the next. 
           Args: 
             list_of_epochs - Entry at index i specifies the number of epochs to train the i-th autoencoder
             switch_fn - A function that switches to the next autoencoder 
        '''
        def __init__(self, list_of_epochs, switch_fn):
            super().__init__()
            self.switch_fn = switch_fn
            self.list_of_epochs = []
            self.cur_idx = 0
            self.last_epoch = 0
            cur_sum = 0
            for epoch in list_of_epochs:
                self.list_of_epochs.append(epoch + cur_sum)
                cur_sum+=epoch
                
        def on_epoch_end(self, epoch):
            epoch += 1
            if epoch == self.list_of_epochs[self.cur_idx] and self.cur_idx < len(self.list_of_epochs) - 1:
                self.cur_idx+=1
                self.switch_fn()
            self.last_epoch = epoch

    def __init__(self, *, input_size:int, hidden_layers:List[int], kl_weight:List[float], 
        activation_fns:List[str], rho=None, batch_norm=None, 
        dropout=None, logger=None, tensorboard_dir=None):

        '''
          Args: 
            input_size - The number of features in the dataset
            hidden_layers - A a list of integers, specifying how many hidden units each autoencoder will have
            rho - A list of floats specifying the desired average activation in each autoencoder
            kl_weight - A list of floats specfiying the strength of the KL-divergence in each autoencoder
            activation_fns - A list of the names of the activation functions to be applied. Must have the same length as hidden_layers
            dropout - Boolean value indicating whether to apply dropout to the hidden layers
            batch_norm - Boolean value indicating whether to apply batch normalization to the hidden layers
            tensorboard_dir - The directory where TensorBoard metrics will be stored
            logger - A logger to keep track of training progress and other information
        ''' 

        super().__init__(
            input_size=input_size, output_size=input_size, 
            neurons_per_layer=hidden_layers, 
            activation_fns=activation_fns, logger=None
        )
            
        self.logger = logger
        
        self.dropout=False if dropout is None else dropout
        self.batch_norm = False if batch_norm is None else batch_norm

        self.switch_callback = None
        self.hidden_layers = hidden_layers
        self.autoencoders = []
        self.cur_idx = 0

        self.kl_weight = kl_weight
        self.rho = rho

        for i, hidden_layer_size in enumerate(hidden_layers):
            if tensorboard_dir:
                cur_tensorboard_dir = tensorboard_dir + '_' + str(i)
            else: cur_tensorboard_dir = None
            autoencoder = SparseAutoencoder(
                input_size=input_size,num_hidden_neurons=hidden_layer_size,
                kl_weight=self.kl_weight[i], rho=self.rho[i], batch_norm=batch_norm, 
                dropout=dropout, activation_fn=activation_fns[i], 
                logger=logger, 
                tensorboard_dir=cur_tensorboard_dir
            )
            self.autoencoders.append(autoencoder)
            input_size = hidden_layer_size

        self._register_metric('kl', lambda o, l, m, p: p['kl'])
        self._register_metric('mse', lambda o, l, m, p: p['mse'])

    def _network_pass(self, data, mode, labels=None):
        return self.autoencoders[self.cur_idx]._network_pass(data,mode,labels)
        
    def build_transformation_fn(self,weights_and_biases, activation_fns):
            if not weights_and_biases: return None

            def transformation_fn(x):
                with torch.set_grad_enabled(False):
                    for (weight, bias), activation_fn in zip(weights_and_biases, activation_fns):
                        x = activation_fn(F.linear(x, weight, bias))
                    return x

            return transformation_fn

    def get_autoencoder(self,index):
        return self.autoencoders[index]

    def _get_model(self):
        return self.autoencoders[self.cur_idx]._get_model()

    def __str__(self):
        first = self.input_size
        msg = ''
        params = []

        previous_l = []
        for i, layer in enumerate(self.hidden_layers):
            for previous in previous_l:
                msg += '({})==>'
                params.append(previous)

            msg += '({})-->[({})->({})'
            params.extend([first, layer, self.activation_fn_names[i]])

            if self.dropout: 
                msg += '->({})'
                params.append('do')

            if self.batch_norm:
                msg += '->({})'
                params.append('bn')

            msg += ']-->({})\n'
            params.append(first)
            previous_l.append(first)
            first = layer
        msg+='Trainable: {}'.format(sum(
                [ae.get_trainable_params() for ae in self.autoencoders]
            )
        )
        return msg.format(*params)

    def get_weights_and_biases(self):
        weights_l, biases_l = [], []
        for ae in self.autoencoders:
            weights, biases = ae.get_weights_and_biases()
            weights_l.append(weights)
            biases_l.append(biases)
        return weights_l, biases_l 

    def train(self, *, train_ds, list_epochs, learning_rates=[0.001], weight_decay=[0], optimizer=None, log_every_n=None, test_ds=None, metrics=None):
        if not isinstance(list_epochs, list):
            raise ValueError('"list_epochs" must be a list. You provided: {}'.format(type(list_epochs)))

        if not isinstance(learning_rates, list):
            raise ValueError('"learning_rates" must be a list. You provided: {}'.format(type(learning_rates)))

        if not isinstance(weight_decay, list):
            raise ValueError('"weight_decay" must be a list. You provided: {}'.format(type(weight_decay)))

        self.learning_rates = learning_rates
        self.weight_decay = weight_decay
        self.list_epochs = list_epochs

        self.weights_and_biases = []
        self.act_fns_temp = []

        self.cur_idx = 0
        def switch_autoencoders():
            self.cur_idx+=1

            old_ae = self.autoencoders[self.cur_idx-1]
            new_ae = self.autoencoders[self.cur_idx]
            self.weights_and_biases.append(old_ae.get_weights_and_biases())
            self.act_fns_temp.append(old_ae.activation_fn)
            new_ae.set_transformation_fn(self.build_transformation_fn(self.weights_and_biases, self.act_fns_temp))

            new_ae._build_optimizer(self.learning_rates[self.cur_idx], weight_decay=self.weight_decay[self.cur_idx], name=optimizer)
            
            if self.logger:
                self.logger.info('Switching to autoencoder {} of {}'.format(self.cur_idx+1, len(self.autoencoders)))

        callback_name = 'switch'
        if self._get_internal_callback_by_name(callback_name):
            self._delete_internal_callback(callback_name)

        self.switch_callback = self.SAECallback(self.list_epochs, switch_autoencoders)
        self._register_internal_callback(self.switch_callback, name=callback_name)

        self.autoencoders[0]._build_optimizer(self.learning_rates[0], weight_decay=self.weight_decay[0], name=optimizer)
        super().train(
            train_ds=train_ds,
            num_epochs=sum(self.list_epochs),
            learning_rate=self.learning_rates[0],
            weight_decay=self.weight_decay[0],
            optimizer=optimizer,
            log_every_n=log_every_n,
            test_ds=test_ds,
            metrics=metrics
        )