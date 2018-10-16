from collections import ChainMap
from functools import reduce
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F

from tensorboardX import SummaryWriter

from PyTorch.callback import EstimatorCallback
import util

from abc import ABC, abstractmethod

class AbstractEstimator():
    DEFAULT_LOG_EVERY_N = 200

    TRAIN = 'train'
    EVAL = 'evaluation'
    PRED = 'predict'
    CUSTOM_FN = 'fn'
    MODE = 'mode'

    PARAMS = 'params'
    LOSS = 'loss'
    OUT = 'output'

    ACTIVATION_FN_DICT = {
        'sigmoid':F.sigmoid,
        'relu':F.relu
    }

    OPTIMIZER_DICT = {
        'adam': optim.Adam,
        'adamax': optim.Adamax,
        'sgd': optim.SGD
    }

    class EarlyStoppingCallback(EstimatorCallback):
        '''Interrupts the training process if after some number of epochs, no improvement in some metric is observed.'''

        def __init__(self, stop_training_fn, mode, patience, get_metric_fn, logger=None):
            super().__init__()
            self.stop_training_fn = stop_training_fn
            self.mode = mode
            self.patience = patience
            self.get_metric_fn = get_metric_fn
            self.last_epoch = 0
            self.cur_best = math.inf if mode == 'min' else -math.inf
            self.patience = patience
            self.waiting = 0
            self.logger=logger

        def on_epoch_end(self, epoch):
            # Guard to ignore evaluation loop
            if epoch > 0 and not (epoch <= self.last_epoch):
                self.last_epoch = epoch

                new_value = self.get_metric_fn()
                if (self.mode=='min' and new_value < self.cur_best) or \
                   (self.mode=='max' and new_value > self.cur_best):
                    self.cur_best = new_value
                    self.waiting = 0
                else: self.waiting += 1

                if self.waiting == self.patience: 
                    if self.logger: self.logger.info('No improvement after {} epochs. Stopping training.'.format(self.patience))
                    self.stop_training_fn()

    class CheckpointCallback(EstimatorCallback):
        def __init__(self, metric_name, path, mode, get_metric_fn, model, logger=None):
            '''Saves checkpoints when a metric has improved in evaluation mode.
              Args:
                path - the full path a file with a .pt extension
                mode - one of 'min', 'max' - 'min' denotes that the smaller values are better and 'max' the opposite
                get_metric_fn - function that returns the current value of the metric
                model - the model that will be saved
            '''
            super().__init__()
            self.get_metric_fn = get_metric_fn
            self.metric_name = metric_name
            self.model = model
            self.mode = mode
            self.last_epoch = 0
            self.current = float('inf') if mode=='min' else float('-inf')

            path, filename = os.path.split(path)
            self.path = os.path.join(path, 'epoch_{}_' + filename[:filename.index('.pt')] + '_{}_{:.4f}.pt')
            
            self.logger = logger

        def on_epoch_end(self, epoch): 
            epoch+=1
            if epoch > self.last_epoch:
                self.last_epoch = epoch
                metric = self.get_metric_fn()
                if self.mode == 'min' and metric < self.current or \
                self.mode == 'max' and metric > self.current: 
                    self.current = metric
                    save_path = self.path.format(epoch, self.metric_name, metric)
                    torch.save(self.model.state_dict(), save_path)

                    if self.logger: self.logger.debug('Saving model to: {}'.format(save_path))

    class ReduceLROnPlateau(EstimatorCallback):
        '''If a metric does not improve after some number of epochs, the learning rate is reduced by some amount.'''

        def __init__(self, get_optimizer_fn, mode, factor, patience, get_metric_fn, get_lr_fn, logger=None):
            super().__init__()
            self.get_optimizer_fn = get_optimizer_fn
            self.patience = patience
            self.factor = factor
            self.mode = mode
            self.scheduler = None
            self.get_metric_fn = get_metric_fn
            self.get_lr_fn = get_lr_fn
            self.last_epoch = 0
            self.last_lr = -1
            self.patience = patience
            self.logger=logger

        def on_epoch_start(self, epoch):
            if not self.scheduler:
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.get_optimizer_fn(), mode=self.mode, patience=self.patience, factor=self.factor)

            if epoch > 0 and not (epoch <= self.last_epoch):
                self.scheduler.step(self.get_metric_fn())
                self.last_epoch = epoch

                lr = self.get_lr_fn()
                if lr != self.last_lr:
                    if self.last_lr != -1:
                        msg = 'No improvement in last {} epochs. Reducing learning rate from {} to {}'.format(self.patience, self.last_lr, lr)
                        if self.logger: self.logger.info(msg)
                        else: print(msg)
                    self.last_lr = lr


    def __init__(self, input_size, output_size, activation_fns=None, neurons_per_layer=None,  tensorboard_dir=None, logger=None):

        if not isinstance(activation_fns, list):
            raise ValueError('"activation_fns" must be a list of integers. You provided: {}'.format(type(activation_fns)))

        if neurons_per_layer is not None and not isinstance(neurons_per_layer, list):
            raise ValueError('"neurons_per_layer" must be a list of integers. You provided: {}'.format(type(neurons_per_layer)))

        for act_fn in activation_fns:
            if act_fn not in self.ACTIVATION_FN_DICT:
                raise ValueError('Invalid activation function provided: {}'.format(act_fn))

        if neurons_per_layer is not None and activation_fns is None:
            raise ValueError('No activation functions provided.')

        if neurons_per_layer is not None and len(activation_fns) != len(neurons_per_layer):
            raise ValueError('The number of activation functions ({}) does not match the number of layers. ({})'.format(
                len(activation_fns), len(neurons_per_layer)
            ))

        BEST_VALUES = {
            'accuracy':{'val': 0, 'epoch':0},
            'precision':{'val': 0, 'epoch':0},
            'recall':{'val': 0, 'epoch':0},
            'f_measure':{'val': 0, 'epoch':0},
            'loss':{'val': math.inf, 'epoch':math.inf}
        }

        METRICS = {
            'accuracy':{self.TRAIN:[],self.EVAL:[],self.MODE:'max'},
            'precision':{self.TRAIN:[],self.EVAL:[],self.MODE:'max'},
            'recall':{self.TRAIN:[],self.EVAL:[],self.MODE:'max'},
            'f_measure':{self.TRAIN:[],self.EVAL:[],self.MODE:'max'},
            'loss':{self.TRAIN:[],self.EVAL:[],self.MODE:'min'}
        }

        self.logger = logger

        # Class that is used to write TensorBoard data
        self.summary_writer = None
        if tensorboard_dir is not None:
            self.summary_writer = SummaryWriter(log_dir=tensorboard_dir)

        self.optimizer = None
        self.optimizer_name = None

        self.activation_fn_names = activation_fns
        self.activation_fns = [self.ACTIVATION_FN_DICT[self.activation_fn_names[i]] for i in range(len(activation_fns))]
        
        self.input_size = input_size
        self.output_size = output_size
        self.neurons_per_layer = neurons_per_layer

        self.global_step = 1
        self.cur_train_epoch = 0

        self.available_metrics = ['loss']
        self.default_callback = EstimatorCallback()
        self.internal_callbacks = []
        self.internal_callbacks_by_name = {}

        self.best_values = BEST_VALUES
        self.metrics = METRICS
        self.custom_metrics = {}
        self.all_metrics = ChainMap(self.metrics, self.custom_metrics)

        self.eval_metrics = []

        # A way to control the provided weight-decay, since 
        # it must be divided by two in the autoencoder and
        # in PyTorch weight-decay is specified in the optimizer
        self.weight_decay_factor = 1

        self.fn_dict = {
            self.TRAIN: self._train,
            self.EVAL: self._eval,
            self.PRED: self._predict,
        }

        self.with_gpu = torch.cuda.is_available()

        # Used to interrupt training
        self.keep_training = True 

    ##############################
    ####### Public Methods #######
    ##############################


    def stop_training(self):
        '''Stops the training after the current epoch ends''' 
        self.keep_training = False

    def add_early_stopping(self, metric_name, patience):
        '''Adds early stopping callback to stop the training process
        if no improvement is observed after some number of epochs.

           Args:
             metric_name - name of the metric to be monitored
             patience - how many epochs with no improvement to wait before interrupting training
        '''

        def get_metric_fn(): 
            return self._get_metric(metric_name, mode=self.EVAL)
            
        early_stopping = self.EarlyStoppingCallback(self.stop_training, self.all_metrics[metric_name][self.MODE], patience, get_metric_fn, logger=self.logger)
        self._register_internal_callback(early_stopping)
        
    def add_checkpoint_saver(self, metric_name, save_dir, name_suffix=None):
        '''Adds a callback to save the weights when some metric improves.
           Args:
             metric_name - name of the metric to be monitored
             save_dir - the directory where checkpoints will be stored
             name_suffix - By default the file name is 'model.pt'. If this parameter is set, the name will be model_{name_suffix}.pt instead.
        '''
        filename = 'model.pt'

        util.makedir_maybe(save_dir)

        if name_suffix is not None: 
            filename = 'model_' + name_suffix + '.pt'

        path = os.path.join(save_dir, filename)

        def get_metric_fn(): 
            return self._get_metric(metric_name, mode=self.EVAL)

        checkpoint_saver = self.CheckpointCallback(metric_name, path, self._get_mode(metric_name), get_metric_fn, self._get_model(), logger=self.logger)
        self._register_internal_callback(checkpoint_saver)


    def reduce_lr_on_plateau(self, *, metric_name, factor=0.5, patience=2):
        '''Reduces the learning rate by some factor after no improvement for some number of epochs

          Args: 
            metric_name - A registered metric name. (Use get_available_metrics() for a list thereof)
            factor - The amount by which the learning rate will be reduced. Formula: new_lr = old_lr * factor
            patience - Number of epochs without improvement before the learning rate is to be reduced
        '''
        def get_metric_fn(): 
            return self._get_metric(metric_name, mode=AbstractEstimator.EVAL)

        def get_lr_fn():
            return self._get_learning_rate()

        def get_optimizer_fn():
            return self.optimizer

        scheduler = self.ReduceLROnPlateau(get_optimizer_fn, self.all_metrics[metric_name][self.MODE], factor, patience, get_metric_fn, get_lr_fn, logger=self.logger)
        self._register_internal_callback(scheduler)

    def get_current_best(self):
        return_dict = {}
        if self.cur_train_epoch > 0:
            for metric in self.available_metrics:
                return_dict[metric] = (self.best_values[metric]['val'], self.best_values[metric]['epoch'])
        else: 
            msg = 'Metrics can only be returned once the model has been trained.'
            if self.logger: self.logger.warning(msg)
            else: print(msg)

        return return_dict


    def get_all_metrics(self):
        '''
            Returns a dict of the format: 
            {
                metric_name:{
                    'train':[...],
                    'eval':[...],
                    'best':(value,epoch)
                    }
                }
            }

            Where metric_name is placeholder for all registered metrics with the model, the 
            train/eval keys contain a list of all intermediate results and best contains a tuple
            with the best value for the specified metric and the epoch it was recorded. Best 
            values are only available for values recorded in evaluation mode. 
        '''
        return_dict = {}
        if self.cur_train_epoch > 0:
            for metric in self.available_metrics:
                return_dict[metric] = {}
                return_dict[metric]['train'] = self.all_metrics[metric][self.TRAIN]
                return_dict[metric]['eval'] = self.all_metrics[metric][self.EVAL]
                return_dict[metric]['best'] = (self.best_values[metric]['val'], self.best_values[metric]['epoch'])
        else: 
            msg = 'Metrics can only be returned once the model has been trained.'
            if self.logger: self.logger.warning(msg)
            else: print(msg)

        return return_dict

    def _build_optimizer(self, learning_rate, weight_decay, name=None):
        if name is None: name = 'adam'
        self.optimizer_name = name
        weight_decay *= self.weight_decay_factor
        self.optimizer = self.OPTIMIZER_DICT[name](filter(lambda p: p.requires_grad,self._get_model().parameters()), learning_rate, weight_decay=weight_decay)

    def train(self, *, train_ds, num_epochs, weight_decay=0, learning_rate=0.001, optimizer=None, log_every_n=None, test_ds=None, metrics=None, callbacks=None):
        self._build_optimizer(learning_rate, weight_decay=weight_decay, name=optimizer)
        if metrics: self.available_metrics.extend(metrics)
        self._run_loop(main_ds=train_ds, log_every_n=log_every_n, internal_eval_ds=test_ds, mode=AbstractEstimator.TRAIN, num_epochs=num_epochs,callbacks=callbacks)

    def evaluate(self, dataset, metrics=None, callbacks=None):
        if metrics: self.available_metrics.extend(metrics)
        self._run_loop(main_ds=dataset, mode=AbstractEstimator.EVAL,callbacks=callbacks)

        if metrics: 
            for metric in metrics: 
                self._remove_metric(metric)
                self.available_metrics.remove(metric)

          
    def predict(self, dataset):
        self._run_loop(main_ds=dataset, mode=AbstractEstimator.PRED)

    def load_model(self, path):
        self._get_model().load_state_dict(torch.load(path))

    def get_trainable_params(self):
        '''Returns the number of trainable parameters for the spcified model'''

        return sum(p.numel() for p in self._get_model().parameters() if p.requires_grad)

    @staticmethod
    def get_available_activation_fns():
        return sorted([key for key in AbstractEstimator.ACTIVATION_FN_DICT])

    def get_available_metrics(self):
        '''Returns a list of the names of the available metrics for this model'''
        return self.available_metrics

    def get_weight_and_bias(self, layer):
        '''Returns the weight and bias tensors for the given layer as a tuple'''
        return layer.weight, layer.bias


    ###############################
    ####### Private methods #######
    ###############################

    def _get_mode(self,metric):
        return self.all_metrics[metric][self.MODE]

    def _get_internal_callback_by_name(self,name):
        if name not in self.internal_callbacks_by_name: return None
        return self.internal_callbacks_by_name[name][0]

    def _register_internal_callback(self, callback, name=None):
        self.internal_callbacks.append(callback)
        if name: self.internal_callbacks_by_name[name] = (callback,len(self.internal_callbacks)-1)

    def _delete_internal_callback(self,name):
        _, idx = self.internal_callbacks_by_name[name]
        self.internal_callbacks[idx] = EstimatorCallback()
        del self.internal_callbacks_by_name[name]

    def _get_learning_rate(self):
        '''
          Returns the learning rate of the first element in param_groups. 
          This works, since all of them have the same learning rate.
          https://stackoverflow.com/questions/48324152/pytorch-how-to-change-the-learning-rate-of-an-optimizer-at-any-given-moment-no
        '''
        return self.optimizer.param_groups[0]['lr']

    def _set_available_metrics(self, metrics):
        with_duplicates = self.available_metrics.extend(metrics)
        self.available_metrics = list(set(with_duplicates))
    
    def _set_learning_rate(self, new_learning_rate):
        if self.optimizer:
            for group in self.optimizer.param_groups:
                group['lr'] = new_learning_rate

    def _set_weight_decay(self, new_weight_decay):
        if self.optimizer:
            for group in self.optimizer.param_groups:
                group['weight_decay'] = new_weight_decay

    def _set_weight_and_bias(self, layer, new_weight, new_bias): 
        '''
          Updates the weight and bias of the given layer to new_weight and new_bias. 
          An error is raised if the shapes of the new weight/bias do not match the current ones. 
        '''
        current_weight = layer.weight.data
        current_bias = layer.bias.data
        
        if current_weight.shape != new_weight.shape:
            raise ValueError('Setting weight: required shape: {}  you provided: {}'.format(current_weight.shape, new_weight.shape))

        if current_bias.shape != new_bias.shape:
            raise ValueError('Setting bias: required shape: {}  you provided: {}'.format(current_bias.shape, new_bias.shape))
        
        current_weight.copy_(new_weight) 
        current_bias.copy_(new_bias) 
    

    def _set_weights_and_biases(self, layers, weights, biases, force_equal_length=False):
        '''
          For each index in the list of layers, the weight and bias at 
          the corresponding index in weights and biases are set 
          for that layer. 

          Args:
            layers - the layers for which new weights/biases are to be set
            weights - the weight matrices 
            biases - the bias vectors
            force_equal_length - If set to false and m<n weights are specified and 
            n layers are present, only the first m layers are updated. If more 
            weights/biases than the available layers are spcified, the redundant ones are ignored.
            Setting to true raises an error if the three lists have different length.
        '''
        # if len(layers) != len(weights) or len(weights) != len(biases):
        #     raise ValueError('Parameters "layers", "weights" and "biases" must have the same length. Lengths: {} {} {}'.format(len(layers),len(weights),len(biases)))

        for layer, weight, bias in zip(layers,weights,biases):
            self._set_weight_and_bias(layer,weight,bias)

    def _reduce_mean(self, values):
        if len(values) == 0: return 0
        return reduce(lambda x, y: x + y, values) / len(values)

    def _toggle_layer(self, layer, trainable):
        for param in layer.parameters():
            param.requires_grad = trainable

    def _freeze_layer(self, layer):
        '''Disable training for the specified layer'''
        return self._toggle_layer(layer, False)

    def _unfreeze_layer(self, layer):
        '''Enable training for the specified layer'''
        return self._toggle_layer(layer, True)


    
    ######################################
    ####### Metric-related methods #######
    ######################################

    def _register_metric(self, metric_name, calc_fn, mode='min'):
        '''
          Registers a custom metric to be monitored. 

          Args:
            metric_name - Name that will be used for internal lookup and for logger messages.
            calc_fun - A function that will calculate the metric. Receives the model's output and labels for the current batch, the mode and optionally custom parameters as defined in _network_pass(). Should return a number and not a tensor. 
            mode - one of 'min' or 'max'. Set to 'min' for metrics, where smaller value is better and to 'max' otherwise.
        '''
        if metric_name not in self.custom_metrics:
            self.custom_metrics[metric_name] = {
                AbstractEstimator.TRAIN:[],
                AbstractEstimator.EVAL:[],
                AbstractEstimator.CUSTOM_FN:calc_fn,
                self.MODE:mode
            }
            self.best_values[metric_name] = {'val': 0 if mode == 'max' else math.inf, 'epoch':0}

    def _remove_metric(self, metric_name):
        if metric_name in self.custom_metrics:
            del self.custom_metrics[metric_name]

    def _reset_metrics(self, mode=None):
        for key in self.metrics:
            if not mode or mode == AbstractEstimator.TRAIN:
                self.metrics[key][AbstractEstimator.TRAIN] = []
            if not mode or mode == AbstractEstimator.EVAL:
                self.metrics[key][AbstractEstimator.EVAL] = []

    def _log_metrics(self, mode, step=None, precision=4):
        if self.logger:
            msg = ''
            params = []
            for metric in self.available_metrics: 
                msg += '{}: {}; '
                params.append(metric)
                params.append(round(self._get_metric(metric, mode),precision))
            if step: info = 'Step {} in epoch {}: '.format(step, self.cur_train_epoch)
            else: info = 'Evaluation after epoch {}: '.format(self.cur_train_epoch)
            msg = msg[:-2]
            msg = msg.format(*params)
            if mode == self.EVAL: self.logger.info(info + msg)
            else: self.logger.debug(info + msg)

    def _write_metric_summaries(self, mode): 
        if self.summary_writer:
            for metric in self.available_metrics: 
                latest_value = self.all_metrics[metric][mode][-1]
                self.summary_writer.add_scalar(mode + '/' + metric, latest_value, global_step=self.global_step)

    def _maybe_update_all_best(self):
        values = []
        for metric in self.available_metrics:
            value = self._get_metric(metric, self.EVAL)
            values.append((metric,value))
            self._maybe_update_best(value, metric)

        self.eval_metrics.append(values)

    def _maybe_update_best(self, new_value, metric):
        '''Sets new_value as the best current value for the metric if it is better, otherwise does nothing.'''
        best_val = self.best_values[metric]['val']
        mode = self.all_metrics[metric][self.MODE]
        if new_value > best_val and mode == 'max' or new_value < best_val and mode == 'min':
            self.best_values[metric]['val'] = new_value
            self.best_values[metric]['epoch'] = self.cur_train_epoch

            # if self.logger and metric in self.available_metrics and best_val not in [0, math.inf]:
            #     self.logger.info('Metric {} improved from {:.4f} to {:.4f} in epoch {}'.format(metric,best_val,new_value,self.cur_train_epoch))

    def _calc_arpf(self, output, labels, mode):
        '''
           Calculates accuracy, precision, recall and f-measure for the specified output and labels
           and appends the newly-calculated valuse to the respective history lists.
        ''' 

        _, predicted_labels = torch.max(output, 1)
        
        acc, fp, fn, tp = 0, 0, 0, 0
        for i, prediction in enumerate(predicted_labels):
            if prediction != 0 and labels[i] == 0: # predicted attack, but actually normal
                fp += 1
            elif prediction == 0 and labels[i] != 0: # predicted normal, but actually attack
                fn += 1
            else:
                if prediction == labels[i]:
                    acc += 1
                    if prediction != 0:
                        tp+=1
                elif prediction != 0 and labels[i] != 0:
                    tp+=1

        accuracy = acc / len(labels)
        precision = tp/(tp + fp) if (tp + fp) > 0 else 0
        recall = tp/(tp + fn) if (tp + fn) > 0 else 0
        f_measure = (2 * recall * precision) / (recall + precision) if (recall + precision) > 0 else 0

        self.metrics['accuracy'][mode].append(accuracy)
        self.metrics['precision'][mode].append(precision)
        self.metrics['recall'][mode].append(recall)
        self.metrics['f_measure'][mode].append(f_measure)

    def _log_best(self):
        '''Prints a summary of the best values so far for each monitored metric.'''
        if self.logger:
            params = []
            msg = 'Summary: \n'
            for metric in self.available_metrics:
                params.extend([metric, self.best_values[metric]['val'],self.best_values[metric]['epoch']])
                msg += '\tBest value for {}: {:.4f} in epoch {} \n'

            self.logger.info(msg.format(*params))

        
    def _calc_custom(self, output, labels, mode, params=None):
        '''
         Calculates custom metrics based on the provided function. 

         Args:
            output - The output as returned by _build_network()
            labels - Labels found in the dataset. None if no labels present
            mode - The current mode (training, evaluation or prediction)
            params - Optional parameters returned by _build_network()
        '''
        for key in self.custom_metrics: 
            metric_fn = self.custom_metrics[key][AbstractEstimator.CUSTOM_FN]

            custom_metric = metric_fn(output, labels, mode, params)
            if not isinstance(custom_metric, (int, float, complex)):
                raise ValueError('Custom metric "{}" is of type {}. Only numbers are allowed.'.format(key,type(custom_metric)))

            self.custom_metrics[key][mode].append(metric_fn(output, labels, mode, params))

    def _get_metric(self,metric_name,mode):
        '''Returns the mean of the specified metric. NaN is returned if the metric is not found.'''
        metric = None

        if metric_name in self.all_metrics:
            metric = self.all_metrics[metric_name][mode]
        else:
            if self.logger:
                self.logger.warning('Metric {} is not registered.'.format(metric_name))
            return float('nan')

        return self._reduce_mean(metric)

    def _update_loss(self, loss, mode):
        '''Adds the newly-calculated loss value to the metrics history'''
        self.metrics['loss'][mode].append(loss)


        
    #############################################
    ####### Training loop-related methods #######
    #############################################

    def _run_loop(self, main_ds, mode, internal_eval_ds=None, num_epochs=None, callbacks=None, log_every_n=None, internal_evaluation=False):
        calculate_every_n = 10 if internal_evaluation else 100

        if not num_epochs:
            num_epochs = 1

        if log_every_n is None:
            log_every_n = AbstractEstimator.DEFAULT_LOG_EVERY_N

        if not callbacks:
            callbacks = [self.default_callback]

        if self.internal_callbacks:
            callbacks.extend(self.internal_callbacks)

        step = 1
        if internal_evaluation: 
            self._reset_metrics(mode=mode)

        for callback in callbacks:
            callback.before_run()

        for epoch in range(num_epochs):
            if mode == AbstractEstimator.TRAIN:
                self.cur_train_epoch += 1
            
            if not internal_evaluation:
                for callback in callbacks:
                    callback.on_epoch_start(epoch)

            for batch in main_ds:

                for callback in callbacks:
                    callback.on_batch_start(batch)

                data = batch['data']

                labels = None
                if 'label' in batch: labels = torch.squeeze(batch['label'], dim=1)

                if self.with_gpu:
                    data = data.cuda()
                    labels = labels.cuda()

                result_dict = self.fn_dict[mode](data, labels)
                output = result_dict[AbstractEstimator.OUT]
                loss, params = None, None

                if mode != AbstractEstimator.PRED:
                    loss = result_dict[AbstractEstimator.LOSS]
                
                if AbstractEstimator.PARAMS in result_dict:
                    params = result_dict[AbstractEstimator.PARAMS]

                if mode != AbstractEstimator.PRED:
                    if step % calculate_every_n == 0:
                        if labels is not None: self._calc_arpf(output, labels, mode)
                        self._calc_custom(output, labels, mode, params=params)
                        self._update_loss(loss.item(), mode)

                    if log_every_n > 0 and step % log_every_n == 0 and not internal_evaluation:
                        self._log_metrics(step=step, mode=mode)
                        self._write_metric_summaries(mode=mode)
                else: pass # Sorry no time to implement prediction mode :/
                    
                for callback in callbacks:
                    callback.on_batch_end(batch)

                if mode == self.TRAIN:
                    self.global_step += 1 
                step+=1 
            
            for callback in callbacks:
                callback.before_evaluation(epoch)

            if mode == AbstractEstimator.TRAIN and internal_eval_ds:
                self._internal_eval(internal_eval_ds)

            if not internal_evaluation:
                for callback in callbacks:
                    callback.on_epoch_end(epoch)

                if not self.keep_training: 
                    if self.logger: self.logger.info('Training interrupted.')
                    break
 
        if internal_evaluation:
            if self.logger:
                self._log_metrics(mode=mode)
            self._write_metric_summaries(mode=mode)
            self._maybe_update_all_best()

        for callback in callbacks:
            callback.on_run_end()

        if mode == self.TRAIN:
            self._log_best()

    def _eval(self, data, labels=None):
        '''Called from _run_loop when in evaluation mode'''
        return self._network_pass(data, labels=labels, mode=AbstractEstimator.EVAL)

    def _train(self, data, labels=None):
        '''Called from _run_loop when in training mode'''
        return self._network_pass(data, labels=labels, mode=AbstractEstimator.TRAIN)

    def _predict(self, data, labels=None):
        '''Called from _run_loop when in prediction mode'''
        return self._network_pass(data, labels=labels, mode=AbstractEstimator.PRED)

    def _internal_eval(self, dataset):
        '''Called when evaluation is performed after a training epoch'''

        self._run_loop(main_ds=dataset, mode=AbstractEstimator.EVAL, internal_evaluation=True)


    def get_name(self):
        return ''
        
    ##################################
    ######## Abstract Methods ########
    ##################################
    
    @abstractmethod
    def _get_model(self):
        '''Returns the object that implements the layers and the forward() function'''
        pass

    @abstractmethod
    def __build_prediction(self):
        '''Converts the output into a prediction that is returned in prediction mode.'''
        pass

    @abstractmethod
    def __loss(self):
        '''Defines the loss function that will be optimized'''
        pass

    @abstractmethod
    def _network_pass(self, data, mode, labels=None): 
        '''
          This function will be called inside of the training/evaluation/prediction loop. 
          
          Args:
            data - a single batch of data as given by the dataset provided to the model
            labels - a single batch of lables if provided to the model
            mode - the current mode (AbstractEstimator.TRAIN, AbstractEstimator.EVAL or AbstractEstimator.PREDICT) 

          Returns: 
            A dict in the form {'output': ... , 'loss': ... , 'params': ...}. Loss is to be specified only in training and evaluation mode. The 'params' key is always optional and will be passed to any custom metrics functions. 
        '''
        pass