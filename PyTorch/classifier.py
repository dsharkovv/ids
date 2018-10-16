import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F

from typing import List
import util

from tensorboardX import SummaryWriter

from PyTorch.abstract_estimator import AbstractEstimator

class Classifier(AbstractEstimator):
    class Model(nn.Module):
        def __init__(self, input_size, hidden_layers, output_size, 
        dropout, batch_norm, layers_to_freeze=None, activation_fns=None):
            super().__init__()

            self.dropout = dropout
            self.batch_norm = batch_norm

            self.activation_fns = activation_fns
            self.hidden_layer_neurons = hidden_layers 
            self.hidden_layers = []
            self.dropout_layers = []
            self.batch_norm_layers = []

            self.layers_to_freeze = layers_to_freeze

            for num_neurons in self.hidden_layer_neurons:
                layer = nn.Linear(input_size, num_neurons)
                self.hidden_layers.append(layer)

                if self.batch_norm:
                    self.batch_norm_layers.append(torch.nn.BatchNorm1d(num_neurons))

                if self.dropout:
                    dropout_layer = nn.Dropout(p=0.25)
                    self.dropout_layers.append(dropout_layer)

                input_size = num_neurons
            self.hidden_layers = nn.ModuleList(self.hidden_layers)
            if self.dropout: self.dropout_layers = nn.ModuleList(self.dropout_layers)
            if self.batch_norm: self.batch_norm_layers = nn.ModuleList(self.batch_norm_layers)

            if len(self.hidden_layer_neurons) > 0:
                n_neurons = self.hidden_layer_neurons[-1]
            else: 
                n_neurons = input_size
            self.output_layer = nn.Linear(n_neurons, output_size)

        def forward(self, x):
            inputs = x
            for i, hidden_layer in enumerate(self.hidden_layers):
                
                # Pass through activation function if one is given
                if self.activation_fns[i] is not None:
                    activation = self.activation_fns[i](hidden_layer(inputs))
                else:
                    activation = hidden_layer(inputs)

                if not self.layers_to_freeze or \
                  (self.layers_to_freeze and not i in self.layers_to_freeze):
                    if self.batch_norm:
                        activation = self.batch_norm_layers[i](activation)

                    if self.dropout:
                        activation = self.dropout_layers[i](activation)
                    
                inputs = activation

            return self.output_layer(inputs)

    def _get_model(self):
        return self.model

    def __init__(self, *, input_size:int, num_classes:int, 
    layers_to_freeze=None, tensorboard_dir=None, batch_norm:bool=None, 
    dropout:bool=None, train_last_only=None, hidden_layers:List[int]=None, 
    activation_fns:List[str]=None, weights=None, biases=None, logger=None):

        '''
          Args: 
            input_size - The number of features in the dataset
            hidden_layers - A list with the number of hidden units in each layer. The length of the list
            is the total number of hidden layers.
            num_classes - Number of classes in the dataset. Determines the number of units in the output layer.
            weights - A list of weights to be used for initialization. Must have the same length as hidden_layers
            biases - A list of biases to be used for initialization. Must have the same length as hidden_layers
            activation_fns - A list of the names of the activation functions to be applied. Must have the same length as hidden_layers
            dropout - Boolean value indicating whether to apply dropout to the hidden layers
            batch_norm - Boolean value indicating whether to apply batch normalization to the hidden layers
            tensorboard_dir - The directory where TensorBoard metrics will be stored
            train_last_only - Disables training on all layers, but the last
            logger - A logger to keep track of training progress and other information
        '''

        super().__init__(
            input_size=input_size, output_size=num_classes, 
            neurons_per_layer=hidden_layers, activation_fns=activation_fns,
            tensorboard_dir=tensorboard_dir, logger=logger
        )

        if(weights is not None and biases is None) or \
          (weights is None and biases is not None):
            raise ValueError('Either no weights and no biases or both weights and biases have to be specified.')

        if weights is not None and biases is not None and \
           (len(hidden_layers) != len(weights) or \
           len(weights) != len(biases)):
            raise ValueError('Provided number of weights ({}) or biases ({}) does not match the number of layers. ({})'.format(  
                    len(weights),len(biases),len(hidden_layers)
                )
            ) 

        if not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError('"num_classes" must be a non-zero positive integer. You provided: {}'.format(num_classes)) 
        
        self.dropout=False if dropout is None else dropout
        self.batch_norm = False if batch_norm is None else batch_norm
        self.train_last_only = False if train_last_only is None else train_last_only

        self.num_classes = num_classes
        self.hidden_layers = hidden_layers
        if self.hidden_layers is None: self.hidden_layers = []

        self.logger = logger
        torch.set_default_tensor_type(torch.DoubleTensor)

        self.model = self.Model(
            self.input_size, self.hidden_layers, self.num_classes, 
            batch_norm=self.batch_norm, dropout=self.dropout, 
            activation_fns=self.activation_fns,
            layers_to_freeze=layers_to_freeze
        )

        if self.with_gpu: self.model = self.model.cuda()

        if weights is not None and biases is not None:
            if self.logger: self.logger.info('Loading weights...') 
            self._set_weights_and_biases(self.model.hidden_layers, weights, biases)

        if layers_to_freeze is not None:
            for idx in layers_to_freeze:
                self._freeze_layer(self.model.hidden_layers[idx])

        if train_last_only:
            if self.logger: self.logger.info('Training last layer only.')
            for layer in self.model.hidden_layers:
                self._freeze_layer(layer)
        
    def __str__(self): 
        msg = '({})-->'
        params = [self.input_size]

        for i, layer in enumerate(self.hidden_layers):
            msg += '[({})->({})'
            params.extend([layer, self.activation_fn_names[i]])

            if self.dropout: 
                msg += '->({})'
                params.append('do')

            if self.batch_norm:
                msg += '->({})'
                params.append('bn')

            msg += ']-->'
        msg+='({}) '.format(self.num_classes)
        msg+='Trainable: {}'.format(self.get_trainable_params())
        return msg.format(*params)

    def _loss(self, output, labels):
        loss = F.cross_entropy(input=output, target=labels)
        if self.with_gpu: loss = loss.cuda()
        return loss

    def _network_pass(self, data, labels, mode):
        '''
          Implementation of abstract method. Performs backpropagation if in training mode and only a forward-pass in evaluation mode
        ''' 
        is_in_train_mode = mode == self.TRAIN

        torch.set_grad_enabled(is_in_train_mode)
        self.model.train(is_in_train_mode)

        output = self.model(data)
        loss = self._loss(output=output, labels=labels)
            
        if mode == AbstractEstimator.TRAIN:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {"output":output, "loss":loss}