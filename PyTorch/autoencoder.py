from functools import partial

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from PyTorch.abstract_estimator import AbstractEstimator
        
class SparseAutoencoder(AbstractEstimator):

    '''
    Implementation of a sparse autoencoder that uses KL-divergence to 
    enforce activation sparsity and weight-decay for regularization
    '''

    RHO = torch.tensor([0.05],dtype=torch.float64)

    class Model(nn.Module):
        def __init__(self, input_size, num_hidden_neurons, dropout, batch_norm, activation_fn=F.sigmoid):
            super().__init__()

            self.dropout = dropout
            self.batch_norm = batch_norm

            self.activation_fn=activation_fn
            self.hidden_layer = nn.Linear(input_size, num_hidden_neurons)
            self.output_layer = nn.Linear(num_hidden_neurons, input_size)

            if self.batch_norm: self.batch_norm_layer = torch.nn.BatchNorm1d(num_hidden_neurons)
            if self.dropout: self.dropout_layer = nn.Dropout(p=0.25)

            self.hidden_activation = None
            self.transformation_fn = None # Can only be set at training start

        def forward(self, x):

            if self.transformation_fn is not None:
                x = self.transformation_fn(x)

            self.hidden_activation = self.activation_fn(self.hidden_layer(x))

            if self.batch_norm:
                self.hidden_activation = self.batch_norm_layer(self.hidden_activation)

            if self.dropout:
                self.hidden_activation = self.dropout_layer(self.hidden_activation)

            return F.sigmoid(self.output_layer(self.hidden_activation))


    def __init__(self, *, input_size:int, num_hidden_neurons:int, 
        kl_weight:float, rho:float=None, batch_norm:bool=None, 
        tensorboard_dir:str=None, dropout:bool=None, 
        activation_fn:str='sigmoid', logger=None):

        '''
          Args: 
            input_size - The number of features in the dataset
            num_hidden_neurons - An integer specifying the number of units in the hidden layer
            kl_weight - A float specfiying the strength of the KL-divergence term
            rho - A float specifying the desired average activation of the hidden layer
            activation_fn - The name of the activation function to be applied
            dropout - Boolean value indicating whether to apply dropout to the hidden layers
            batch_norm - Boolean value indicating whether to apply batch normalization to the hidden layers
            tensorboard_dir - The directory where TensorBoard metrics will be stored
            logger - A logger to keep track of training progress and other information
        '''

        super().__init__(
            input_size=input_size, output_size=input_size, 
            neurons_per_layer=[num_hidden_neurons],
            activation_fns=[activation_fn],
            logger=logger, tensorboard_dir=tensorboard_dir
        )

        self.logger = logger
        self.weight_decay_factor = 0.5

        torch.set_default_tensor_type(torch.DoubleTensor)

        if rho is None: self.rho = self.RHO
        else: self.rho = torch.tensor([rho],dtype=torch.float64)
        
        self.dropout=False if dropout is None else dropout
        self.batch_norm = False if batch_norm is None else batch_norm

        self.activation_fn_name = activation_fn
        self.activation_fn = self.activation_fns[0]
        self.num_hidden_neurons = self.neurons_per_layer[0]

        self.model = self.Model(input_size, num_hidden_neurons, dropout, batch_norm, activation_fn=self.activation_fn)
        if self.with_gpu: self.model = self.model.cuda()

        self.kl_weight = kl_weight
        
        self._register_metric('kl', lambda o, l, m, p: p['kl'], mode='min')
        self._register_metric('mse', lambda o, l, m, p: p['mse'], mode='min')

    def __str__(self):
        msg = '({})-->[({})->({})'

        params = [
            self.input_size, 
            self.num_hidden_neurons, 
            self.activation_fn_name
        ]

        if self.dropout: 
            msg += '->({})'
            params.append('do')

        if self.batch_norm:
            msg += '->({})'
            params.append('bn')

        msg += ']-->({})'
        params.append(self.input_size)
        msg+='Trainable: {}'.format(self.get_trainable_params())
        return msg.format(*params)

    def get_name(self):
        return 'ae_in{}_h{}_rho{}'.format(self.input_size, self.num_hidden_neurons,self.rho.item())

    def set_transformation_fn(self, transformation_fn):
        self.model.transformation_fn = transformation_fn

    def _get_model(self):
        return self.model

    def _loss(self, network_output, expected_value):
        mse_loss = F.mse_loss(network_output, expected_value)
        if self.with_gpu: mse_loss.cuda()
        kl_loss = self._kl(self.rho, torch.mean(self.model.hidden_activation))
        return 0.5 * mse_loss + self.kl_weight * kl_loss, mse_loss, kl_loss

    def _kl(self, rho, rho_hat):
        if self.with_gpu: rho = rho.cuda()
        return rho * torch.log(rho) - rho * torch.log(rho_hat) + (1 - rho) * torch.log(1 - rho) - (1 - rho) * torch.log(1 - rho_hat)

    def get_weights_and_biases(self):
        return super().get_weight_and_bias(self.model.hidden_layer)

    def _network_pass(self, data, mode, labels=None):
        
        is_in_train_mode = mode == self.TRAIN
        
        torch.set_grad_enabled(is_in_train_mode)
        self.model.train(is_in_train_mode)

        if self.model.transformation_fn:
            labels = self.model.transformation_fn(data)
        else: labels = data

        output = self.model(data)

        loss, mse, kl = self._loss(output, labels)
        
        if mode == self.TRAIN:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return {"output":output, "loss":loss, "params":{"mse":mse.item(), "kl":kl.item()}}
    