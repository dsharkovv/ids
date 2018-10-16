
import sys
import os
import math
import pickle
import json
from collections import deque, ChainMap
from datetime import datetime
from copy import deepcopy
from functools import partial 
from typing import List, Any, Dict, Callable
from random import shuffle
from sklearn.utils import shuffle as shuffle_sk

import util

import PyTorch.dataset_loader as py_data_loader
from PyTorch.autoencoder import SparseAutoencoder as PY_Autoencoder
from PyTorch.stacked_autoencoder import StackedAutoencoder as PY_StackedAutoencoder
from PyTorch.classifier import Classifier as PY_Classifier
from PyTorch.abstract_estimator import AbstractEstimator

import Tensorflow.dataset_loader as tf_data_loader
from Tensorflow.sparse_autoencoder import SparseAutoencoder as TF_Autoencoder
from Tensorflow.stacked_autoencoder import StackedAutoencoder as TF_StackedAutoencoder
from Tensorflow.classifier import Classifier as TF_Classifier

import torch
import tensorflow as tf

class RunLoop():

    class ModelCache():
        def __init__(self, model, param_dict):
            self.model = model
            self.param_dict = param_dict
            # self.best_metrics = autoencoder.get_current_best()

        @property
        def info(self):
            return self.model.__str__()

    TF_BACKEND = 'tf'
    PY_BACKEND = 'py'

    # keywords
    GRID_SEARCH = 'gs'
    AE, SAE, CL = 'ae', 'sae', 'cl'
    LOAD, QUIT = 'load', 'quit'
    BACK, HELP = 'back', 'help'

    #Defaults
    BATCH_SIZE = 128
    HIDDEN_NEURONS = 200
    ACTIVATION_FN = 'sigmoid'
    RHO, KL = 0.05, 1
    WD = 0.0001
    N_EPOCHS = 10
    LR = 0.001

    # Paths PyTorch
    LOGGING_PATH_PY = os.path.join('Logging','PyTorch')
    CHECKPOINT_PATH_PY = os.path.join('Checkpoints','PyTorch')
    SUMMARY_DIR_PY = os.path.join('TensorBoard','PyTorch')

    # Paths TensorFlow
    LOGGING_PATH_TF = os.path.join('Logging','TensorFlow')
    CHECKPOINT_PATH_TF = os.path.join('Checkpoints','TensorFlow')
    SUMMARY_DIR_TF = os.path.join('TensorBoard','TensorFlow')

    # Messages
    INTERACTIVE_MODE_MSG = '\nRunning in interactive mode. Run with --config <CONFIG PATH> to use a JSON configuration.\n\n'
    DATASET_LOAD_MSG = 'Loading KDD-Dataset. Please wait.'
    DATASET_LOAD_FIN_MSG = 'Loading finished in {:.2f}s'
    BACKEND_MSG = 'Select backend.\nType "tf" for TensorFlow or "py" for PyTorch and press ENTER.'
    LOAD_DS_MSG = 'Select which dataset to load.\nEnter:\n   1 for full\n   2 for two-class\n   3 for five-class'
    CONFIG_MSG = 'The next prompts will let you configure the {}. Enter custom values or press ENTER for deafults.'
    UNKNOWN_COMMAND = '"{}" is an invalid command. Type "' + HELP + '" to get list of available commands.'
    COMMAND_ONLY_INTERACTIVE = 'Command {} is only available in interactive mode. Skipping.'

    # Prompts
    QUIT_PROMPT = 'Are you sure you want to quit? (y/n)'
    AE_PROMPT = 'Would you like to use an existing autoencoder? (y/n)'
    CL_PROMPT = 'Would you like to use an existing classifier? (y/n)'
    AE_PARAMS = 'Would you like to use parameters from an existing autoencoder? (y/n)'
    LR_DECR = 'Enable learning rate decrease if no improvement? (y/n)'

    # Errors
    CONFIG_BACKEND_ERR = 'Error in config: "backend" can be one of "'+ TF_BACKEND + '" or "' + PY_BACKEND + '". You provided "{}"'
    CONFIG_DATASET_ERR = 'Error in config: "dataset" must be one of "full", "2c" or "5c". You provided: {}'
    CONFIG_MODEL_ERR = 'Error in config: "model" must be one of "'+ AE + '", "'+ SAE + '" or "'+ CL + '". You provided: {}' 
    CONFIG_COMMANDS_ERR = 'Error in config: "commands" was not found.'
    CONFIG_PARSE_ERR = 'An error occured while parsing:\n{}'

    # Help messages
    GENERAL_HELP = 'Available commands in interactive mode:\n"cl" - creates a classifier\n"ae" - creates an autoencoder\n' + \
    '"sae" - creates a stacked autoencoder\n"quit" - exits the program\n"back" - cancels configuration prompts\n"load" - changes the dataset\n"help" - returns this message\n\nFurther instructions will be presented as you go.\n'

    def __init__(self, config_path):
        self.date_string = util.get_date_string()
        self.logger = None
        self.interactive_mode = True
        self.grid_search_keys = None
        self.config = {}
        self.dataset = {}
        self.backend = None
        self.dataset_name = ['','full','two-class','five-class'] 
        self.cur_config_command = {}
        self.cur_model_dir = ''
        
        self.s_autoencoders = []
        self.autoencoders = []
        self.classifiers = []
        
        self.available_aes = lambda: len(self.autoencoders) > 0 or len(self.s_autoencoders) > 0
        self.num_grid_searches = 0
        self.cur_main_dir = None

        def check_list_json(the_list, check_fn):
            if not isinstance(the_list,list): 
                if isinstance(the_list,str):
                    the_list = the_list.replace(' ', '')
                    the_list = the_list.split(',')
                else: return None

            converted = [check_fn(val) for val in the_list]
            if any([x is None for x in converted]): return None

            return converted

        def convert_if_string(value, convert_fn):
            if isinstance(value, list) or \
               isinstance(value, dict):
                return None
            
            return convert_fn(value)

        # used in interactive mode
        convert_to_activation_list = partial(util.convert_to_list, map_fn=self.check_if_valid_activation_fn)

        # used with a config file
        toa = self.check_if_valid_activation_fn
        toal = lambda the_list: check_list_json(the_list,self.check_if_valid_activation_fn)
        identity = lambda x: x
        to_bool = lambda x: x if x == True or x == False else None

        from_zero_int = lambda x: convert_if_string(x,lambda y:util.convert_to_int(y, min_value=0.0))
        from_zero_int_l = lambda the_list: check_list_json(the_list,from_zero_int)

        from_zero_float = lambda x: convert_if_string(x,lambda y:util.convert_to_float(y, min_value=0.0))
        from_zero_float_l = lambda the_list: check_list_json(the_list,from_zero_float)

        zero_to_one_float = lambda x: convert_if_string(x,lambda y:util.convert_to_float(y, min_value=0.0, max_value=1.0))
        zero_to_one_float_l = lambda the_list: check_list_json(the_list,zero_to_one_float)

        es_check = lambda x: convert_if_string(x,lambda y:util.convert_to_int(y, min_value=0))
        cross_val_check = lambda x: convert_if_string(x,lambda y:util.convert_to_int(y, min_value=1))

        def rlr_conv(val):
            factor_conv = convert_if_string(val['factor'],lambda y:util.convert_to_float(y, min_value=0.0, max_value=1.0))
            patience_conv = convert_if_string(val['patience'],lambda y:util.convert_to_int(y, min_value=0))

            if factor_conv is None or patience_conv is None:
                return None

            return {'factor': factor_conv, 'patience': patience_conv}

        self.default_params = {
            'bs':self.BATCH_SIZE,'lr':self.LR,
            'kl':self.KL,'wd':self.WD,
            'hn':self.HIDDEN_NEURONS,
            'ne':self.N_EPOCHS,
            'rho':self.RHO,'af':self.ACTIVATION_FN,
            'do':False,'bn':False,'last_only': False,
            'reduce_lr':None,'cs':False,'es':None, 'cross_val':0
        }

        self.param_name = {
            'bs':'batch size','lr':'learning rate',
            'kl':'KL-divergence','wd':'weight decay',
            'hn':'hidden neurons',
            'ne':'number of epochs',
            'rho':'rho','af':'activation function',
            'do':'dropout','bn':'batch normalization','last_only':'disable first',
            'reduce_lr':'reduce learning rate','cs':'checkpoint saver',
            'es':'early stopping', 'cross_val':'cross-validation'
        }
        self.options = ['cross_val','es','lrl','cs','last_only']

        valid_commands = [key for key in self.default_params]
        if config_path:    
            self.config = json.load(open(config_path))

            self.check_missing_attr(self.config,['backend','dataset','commands'])

            if self.config['backend'] not in [self.PY_BACKEND,self.TF_BACKEND]:
                raise ValueError('Invalid backend: {}'.format(self.config['backend']))
            
            self.backend = self.config['backend']

            if self.backend == self.PY_BACKEND: self.logger = util.setup_logger(self.LOGGING_PATH_PY)
            else: self.logger = util.setup_logger(self.LOGGING_PATH_TF)

            pca = None
            if isinstance(self.config['dataset'],str): dataset_name = self.config['dataset']
            elif isinstance(self.config['dataset'],dict):
                self.check_missing_attr(self.config['dataset'],['name'])
                dataset_name = self.config['dataset']['name']
                pca = self.config['dataset'].get('pca')
                if pca and self.backend == self.TF_BACKEND:
                    raise ValueError('PCA is not supported with TensorFlow backend')
                elif not pca: self.logger.warning('"dataset" given as dict but "pca" not specified.')
            else: raise ValueError('Attribute "dataset" must either be a string or a dict')
                
            self.converters = {
                'hn':{self.SAE:from_zero_int_l,self.AE:from_zero_int,self.CL:from_zero_int_l},
                'bs':{self.SAE:from_zero_int,self.AE:from_zero_int,self.CL:from_zero_int},
                'lr':{self.SAE:from_zero_float_l,self.AE:from_zero_float,self.CL:from_zero_float},
                'wd':{self.SAE:from_zero_float_l,self.AE:from_zero_float,self.CL:from_zero_float},
                'ne':{self.SAE:from_zero_int_l,self.AE:from_zero_int,self.CL:from_zero_int},
                'rho':{self.SAE:zero_to_one_float_l,self.AE:zero_to_one_float,self.CL:identity},
                'kl':{self.SAE:from_zero_float_l,self.AE:from_zero_float,self.CL:identity},
                'af':{self.SAE:toal,self.AE:toa,self.CL:toal},
                'do':{self.SAE:to_bool,self.AE:to_bool,self.CL:to_bool},
                'bn':{self.SAE:to_bool,self.AE:to_bool,self.CL:to_bool},
                'last_only':{self.SAE:to_bool,self.AE:to_bool,self.CL:to_bool},
                'es':{self.SAE:es_check,self.AE:es_check,self.CL:es_check},
                'reduce_lr':{self.SAE:rlr_conv,self.AE:rlr_conv,self.CL:rlr_conv},
                'cs':{self.SAE:to_bool,self.AE:to_bool,self.CL:to_bool},
                'cross_val':{self.SAE:cross_val_check,self.AE:cross_val_check,self.CL:cross_val_check}
            }

            self.logger.info('Parsing commands...')

            for command in self.config['commands']:

                required = ['name', 'model']
                self.check_missing_attr(command,required)

                if command['model'] in ['clsae','clae']: required = ['params_ae','params_cl']
                elif command['model'] in [self.CL,self.AE,self.SAE]: required = ['params']
                else: raise ValueError('Not a valid model "{}" in:\n{}'.format(command['model'], command))
                self.check_missing_attr(command,required)

                if ('options' in command and 'reduce_lr' in command['options']):
                    self.check_missing_attr(command['options']['reduce_lr'],['factor','patience'])
                if ('options_cl' in command and 'reduce_lr' in command['options_cl']):
                    self.check_missing_attr(command['options_cl']['reduce_lr'],['factor','patience'])
                if ('options_ae' in command and 'reduce_lr' in command['options_ae']):
                    self.check_missing_attr(command['options_ae']['reduce_lr'],['factor','patience'])

                if 'dataset' in command and isinstance(command['dataset'],dict): 
                    self.check_missing_attr(command['dataset'],['name'])
                    if 'pca' in command['dataset'] and self.backend == self.TF_BACKEND:
                        raise ValueError('PCA is not supported with TensorFlow backend')
                    elif 'pca' not in command['dataset']: self.logger.warning('"dataset" given as dict but "pca" not specified.')

                command['parsed']  = self.convert_single_command(command)

            self.config['commands'] = deque(self.config['commands'])
            self.interactive_mode = False

            self.dataset = self.load_dataset(self.dataset_name_to_selector(dataset_name), pca=pca) 

        else: print(self.INTERACTIVE_MODE_MSG + self.GENERAL_HELP)
    
        self.params_ae = self.get_default_params('ae')
        self.params_sae = self.get_default_params('sae')
        self.params_cl = self.get_default_params('cl')

        self.fns = {
            self.LOAD:{
                self.TF_BACKEND: self.load_dataset,
                self.PY_BACKEND: self.load_dataset
            },
            self.AE:{
                self.TF_BACKEND: {'build': self.build_ae_tf, 'train': self.train_ae_tf},
                self.PY_BACKEND: {'build': self.build_ae_py, 'train': self.train_ae_py}
            },
            self.CL:{
                self.TF_BACKEND: {'build': self.build_cl_tf, 'train': self.train_cl_tf},
                self.PY_BACKEND: {'build': self.build_cl_py, 'train': self.train_cl_py}
            },
            self.SAE:{
                self.TF_BACKEND: {'build': self.build_sae_tf, 'train': self.train_sae_tf},
                self.PY_BACKEND: {'build': self.build_sae_py, 'train': self.train_sae_py}
            }
        }

        if self.interactive_mode:
            self.config_sae = [
                ('Batch size','bs',from_zero_int,'Integer > 0'),
                ('Hidden neurons','hn',from_zero_int_l,'Comma-separated list of integers > 0'),
                ('Activation function','af',convert_to_activation_list,'Comma-separated list of strings'),
                ('Learning rate','lr',from_zero_float_l,'Comma-separated list of floats > 0'),
                ('KL-divergence strength','kl',from_zero_float_l,'Comma-separated list of floats > 0'),
                ('Desired average activation','rho',zero_to_one_float_l,'Comma-separated list of floats > 0 and < 1'),
                ('Weight decay','wd',from_zero_float_l,'Comma-separated list of floats > 0'),
                ('Number of epochs per layer','ne',from_zero_int_l,'Comma-separated list of integers > 0')
            ]

            self.config_sae_short = [
                ('Weight decay','wd',from_zero_float_l,'Comma-separated list of floats > 0'),
                ('Learning rate','lr',from_zero_float_l,'Comma-separated list of floats > 0'),
                ('Number of epochs per layer','ne',from_zero_int_l,'Comma-separated list of floats > 0')
            ]
            self.config_sae_short_tf = [
                ('Number of epochs per layer','ne',from_zero_int_l,'Comma-separated list of floats > 0')
            ]

            self.config_ae = [
                ('Batch size','bs',from_zero_int,'Integer > 0'),
                ('Hidden neurons','hn',from_zero_int,'Integer > 0'),
                ('Activation function','af',self.check_if_valid_activation_fn,'String'),
                ('Learning rate','lr',from_zero_float,'Integer or float > 0'),
                ('KL-divergence strength','kl',from_zero_float,'Integer or float > 0'),
                ('Desired average activation','rho',zero_to_one_float,'Integer or float > 0 and < 1'),
                ('Weight decay','wd',from_zero_float,'Integer or float > 0'),
                ('Number of epochs','ne',from_zero_int,'Integer > 0')
            ]
            
            self.config_ae_short = [
                ('Weight decay','wd',from_zero_float,'Integer or float > 0'),
                ('Learning rate','lr',from_zero_float,'Integer or float > 0'),
                ('Number of epochs','ne',from_zero_int,'Integer > 0')
            ]

            self.config_ae_short_tf = [
                ('Number of epochs','ne',from_zero_int,'Integer > 0')
            ]

            self.config_cl = [
                ('Batch size','bs',from_zero_int,'Integer > 0'),
                ('Hidden neurons','hn',from_zero_int_l,'Comma-separated list of integers > 0'),
                ('Activation function','af',convert_to_activation_list,'Comma-separated list of strings'),
                ('Learning rate','lr',from_zero_float,'Integer or float > 0'),
                ('Weight decay','wd',from_zero_float,'Integer or float > 0'),
                ('Number of epochs','ne',from_zero_int,'Integer > 0')
            ]

            self.config_cl_from_ae = [
                ('Batch size','bs',from_zero_int,'Integer > 0'),
                ('Weight decay','wd',from_zero_float,'Integer or float > 0'),
                ('Learning rate','lr',from_zero_float,'Integer or float > 0'),
                ('Number of epochs','ne',from_zero_int,'Integer > 0')
            ]

            self.config_cl_short = [
                ('Weight decay','wd',from_zero_float,'Integer or float > 0'),
                ('Learning rate','lr',from_zero_float,'Integer or float > 0'),
                ('Number of epochs','ne',from_zero_int,'Integer > 0')
            ]

            self.config_cl_short_tf = [
                ('Number of epochs','ne',from_zero_int,'Integer > 0')
            ]

        self.programm_loop()

    def convert_single_command(self, command):
        '''
           Converts hyperparameters from JSON file to data that can be
           fed into the model for training. It is assumed that a 
           check has been performed to ensure all required 
           parameters are present.
        '''
        model = command['model']

        # model_1 represents the (stacked) autoencoder when
        # it is used for classifier pretraining and
        # model_2 - the classifier respectively
        model_1, model_2 = model, None
        m1_merged, m2_merged = None, None
        m1_options, m2_options = None, None

        if model in ['clsae', 'clae']:
            model_1, model_2 = model[2:], model[:2]

            # Overwrite default parameters with ones provided in the JSON file
            m2_params = command['params_cl']
            m2_def_params = util.dict_entries_to_list(self.get_default_params(model_2))

            m2_parsed = self.parserV2(m2_params, model_2)
            m2_merged = util.overwrite_dict(m2_def_params,m2_parsed)

            m2_options = command.get('options_cl')
            if m2_options: 
                m2_options = self.parserV2(m2_options, model_2, options=True)

        # Either one of "params_ae" or "params" must be present
        m1_params = command.get('params_ae')
        if m1_params is None: m1_params = command['params']

        # Overwrite default parameters with ones provided in the JSON file
        m1_def_params = util.dict_entries_to_list(self.get_default_params(model_1))
        m1_parsed = self.parserV2(m1_params, model_1)
        m1_merged = util.overwrite_dict(m1_def_params,m1_parsed)

        m1_options = command.get('options_ae')
        if m1_options is None: m1_options = command.get('options')
            
        if m1_options: m1_options = self.parserV2(m1_options, model_1, options=True)
        
        self.grid_search_keys = list(m1_parsed.keys())
        if model_2 is not None: 
            self.grid_search_keys.extend(m2_parsed.keys())
            self.grid_search_keys = list(set(self.grid_search_keys))

        return {
            'model_1':{
                'name':model_1, 
                'params':m1_merged,
                'options':m1_options
            },
            'model_2':{
                'name':model_2, 
                'params':m2_merged,
                'options':m2_options
            },
        }


    def check_missing_attr(self, the_dict, attributes):
        for attr in attributes: 
            if attr not in the_dict: 
                raise ValueError('Error: Required attribute "{}" missing in:\n{}'.format(attr,the_dict))

    def get_build_fn(self, model):
        return self.fns[model][self.backend]['build']

    def get_train_fn(self, model):
        return self.fns[model][self.backend]['train']

    def get_dir_with_suffix(self, path, last_dir):
        full = os.path.join(path, last_dir)
        count = 1
        dir_with_suffix = last_dir
        while not util.makedir_maybe(full):
            dir_with_suffix = last_dir + '_' + str(count)
            full = os.path.join(path, dir_with_suffix)
            count += 1

        return dir_with_suffix

    def add_path_suffix_maybe(self, path, last_dir, only_dir=None):
        full = os.path.join(path, last_dir)
        count = 1

        while not util.makedir_maybe(full):
            dir_with_suffix = last_dir + '_' + str(count)
            full = os.path.join(path, dir_with_suffix)
            count += 1

        if only_dir: return dir_with_suffix
        return full

    def pop_command(self):
        command_deque = self.config['commands']
        if len(command_deque) == 0: return self.QUIT

        self.cur_config_command = command_deque.popleft()
        return self.cur_config_command['name']

    def dataset_name_to_selector(self,name):
        if name == 'full': return 1
        elif name == '2c': return 2
        elif name == '5c': return 3
        else: raise ValueError(self.CONFIG_DATASET_ERR.format(name))

    def question_loop(self, message, allowed, convert_fn=None, end=None, no_quit=False, input_text=None):
        if input_text is None: input_text = ''
        if end is None: end = '\n'

        print(message, end=end)
        while True:
            inner_command = input(input_text)
            if not no_quit and inner_command in [self.QUIT, self.BACK]: 
                return inner_command

            if convert_fn: inner_command = convert_fn(inner_command)

            if inner_command in allowed: return inner_command
            else: print('Please enter a valid option. One of {}'.format(allowed))

    def yes_no_question(self, message):
        return self.question_loop(message, allowed=['y','n'], end=' ', no_quit=True)

    def get_grid_search_train_fn(self, main_model, options=None, classifier=None, classifier_params=None, classifier_options=None):
        '''Creates the function that util.grid_search() calls for each combination of parameters.

            Args: 
                main_model - The type of model to be trained. One of 'cl', 'ae' or 'sae'.
                classifier - Can only be 'cl' for this implementation. Used when a (stacked) autoencoder is applied to a classifier.
                classifier_params - The dictionary of parameters to pass to the grid search function. Each entry is a list of different hyperparameter values to be tested.

            Returns: The training function, expected by util.grid_search()
        '''
        final_results = {}
        for key in self.grid_search_keys:
            final_results[key] = []

        def train_fn(iteration, params, other_params=None):

            if other_params: 
                copy_of_other = other_params.copy()
                del copy_of_other['weights']
                del copy_of_other['biases']
                params['from_ae'] = copy_of_other 
                params['iteration'] = iteration

            params_tuple = [(key,value) for key,value in params.items() if key in self.grid_search_keys]
            if self.logger: self.logger.info('Running train_fn: {}'.format(util.get_config(*params_tuple)))

            params_with_options = params.copy()

            if options is not None:
                params_with_options.update(options)
            
            model_name = main_model  

            models = [] # Holds models from cross-validation
            if main_model == self.CL and params_with_options['cross_val'] > 0: 
                folds = params_with_options['cross_val']
                per_fold = len(self.dataset['train']) // folds
                self.logger.info('Performing {} fold cross-validation.'.format(folds))

                train_data = shuffle_sk(self.dataset['train'])
                for i in range(folds):
                    if other_params:
                        model = self.get_build_fn(model_name)(params_with_options, other_params['weights'], other_params['biases'], force_new_dir=True)
                    else: model = self.get_build_fn(model_name)(params_with_options, force_new_dir=True)

                    if self.backend == self.PY_BACKEND:
                        train_fold = py_data_loader.CustomDataset(train_data[:i*per_fold] + train_data[i*per_fold + per_fold:])
                        test_fold = py_data_loader.CustomDataset(train_data[i*per_fold:i*per_fold + per_fold])
                    # TODO: DOUBLE CHECK
                    else:
                        train_fold = train_data.iloc[:i*per_fold] + train_data.iloc[i*per_fold + per_fold:]
                        test_fold = train_data.iloc[i*per_fold:i*per_fold + per_fold]

                    models.append(self.get_train_fn(model_name)(model, params_with_options, cross_val_dataset={'train':train_fold, 'test':test_fold}))

            else: 
                if other_params:
                    model = self.get_build_fn(model_name)(params_with_options, other_params['weights'], other_params['biases'], force_new_dir=True)
                else: model = self.get_build_fn(model_name)(params_with_options, force_new_dir=True)
                
                model = self.get_train_fn(model_name)(model, params_with_options)

            if self.backend == self.PY_BACKEND: 
                for metric in model.available_metrics:
                    if metric not in final_results:
                        final_results[metric] = []

            # We have been training an autoencoder
            # Pass the parameters from the autoencoder
            # to the classifier and do grid search there
            if classifier is not None: 
                ae_params = {}
                weights, biases = model.get_weights_and_biases()
                ae_params['weights'], ae_params['biases'] = weights, biases
                
                for key in params: 
                    if key in classifier_params: 
                        ae_params['ae_' + key] = [params[key]] 
                    else: ae_params[key] = [params[key]]

                classifier_params['hn'] = [params['hn']]
                classifier_params['af'] = [params['af']]

                new_results = util.grid_search(
                    self.get_grid_search_train_fn(main_model=classifier, options=classifier_options), 
                    classifier_params, other_params=ae_params
                )

                for key in new_results: 
                    if key not in ['weights','biases']:
                        if key not in final_results: final_results[key] = []
                        final_results[key].extend(new_results[key])

            if self.backend == self.PY_BACKEND and classifier is None: 
                metric = 'accuracy' if main_model == self.CL else 'kl'

                # Get the average of results from cross-validation
                if models: 
                    epochs = [model.get_all_metrics()[metric]['best'][1] for model in models]
                    results_list = [models[idx].eval_metrics[epochs[idx] - 1] for idx in range(len(models))]  

                    merged_results = [['', []] for _ in range(len(results_list[0]))]
                    for results in results_list: 
                        for i, (metric, value) in enumerate(results): 
                            merged_results[i][0] = metric
                            merged_results[i][1].append(value)
                        
                    for i in range(len(merged_results)):
                        avg = sum(merged_results[i][1]) / len(merged_results[i][1])
                        merged_results[i][1] = avg
                        
                    results = merged_results

                else: 
                    epoch = model.get_all_metrics()[metric]['best'][1]
                    results = model.eval_metrics[epoch-1]

                for metric, value in results: final_results[metric].append(value)
                for metric, value in params_tuple: final_results[metric].append(value)

                if other_params:
                    for key in other_params:
                        if key not in final_results:
                            final_results[key] = []
                        final_results[key].extend(other_params[key]) 

            return final_results

        return train_fn

    def config_loop(self, params, param_dict): 
        '''
        Called in interactive mode when a model is being created. 
        Lets the user set the model parameters.
        '''
        temp_dict = deepcopy(param_dict)
        ENTER = '<ENTER>'
        for param in params:
            command = ENTER

            name = param[0]
            variable = param_dict[param[1]]
            if isinstance(variable,list): variable = variable[0]

            map_fn = param[2]
            dtype = param[3]
            while command == ENTER or command is None:
                command = input('{}? (Default is {}) '.format(name,variable))

                if command in [self.QUIT, self.BACK]:
                    return command

                # Try to convert to the appropriate type if not an empty command
                if command != '': 
                    command = map_fn(command)
                    if command is None:
                        print('Not a valid value. {} is required. Leave empty and press Enter to use default.'.format(dtype))

            if command == '': 
                temp_dict[param[1]] = map_fn(str(variable))
                print('Using default value: {}'.format(variable))
            else: temp_dict[param[1]] = command

        return temp_dict

    def check_if_valid_activation_fn(self, value):
        available = AbstractEstimator.get_available_activation_fns()
        if value in available: return value
        return None

    def get_default_params(self, model_name):
        params = deepcopy(self.default_params)
        if model_name == 'cl':
            del params['rho']
            del params['kl']
        return params

    def load_dataset(self, selector, pca=None):
        '''
          Loads the data required for the selected dataset
          based on the backend that is used. Selector is a value
          from 1 to 3. 1 for full, 2 for two-class and 3 for five-class.
        '''
        if self.logger and not self.interactive_mode:
            self.logger.info('Loading dataset {}'.format(self.dataset_name[selector]))
        else: print('Loading dataset {}'.format(self.dataset_name[selector]))

        # No feature columns in the case of PyTorch
        train, test, feature_cols, n_classes = None, None, None, None
        if self.backend == self.PY_BACKEND:
            train, test, n_classes = py_data_loader.load_KDD(selector, pca=pca, logger=self.logger)
        else: 
            if pca: raise ValueError('PCA is only available with PyTorch.')
            train, test, feature_cols, n_classes = tf_data_loader.load_KDD(selector)
        
        if self.logger and not self.interactive_mode: self.logger.info('Loading complete.')
        return {
            'train': train, 'test': test, 
            'n_classes': n_classes, 'feature_cols': feature_cols
        }

    def use_existing_model_maybe(self, models: List[ModelCache], prompt=None):
        '''
        Returns: 
            The index of the model to be used, -1 if the user
            indicated that they do not want to use an existing model
            or QUIT/BACK signal if the operation was aborted.
        '''
        if prompt is None: prompt = self.AE_PROMPT

        if len(models) > 0:
            val = self.yes_no_question(prompt)
            if val == 'y': return self.get_existing_model(models)
        return -1

    def get_existing_model(self, models: List[ModelCache]):
        '''
        In interactive mode lets the user select a model from a 
        list of already trained models.
        '''
        if self.interactive_mode:
            print()
            print('Following models are available:')
            for i, model in enumerate(models): 
                print(i, model.model)
                print('-'*20)

            msg = "Enter the number of the model you want to use."
            return self.question_loop(msg, allowed=list(range(len(models))), convert_fn=util.convert_to_int)
        else:
            self.logger.warning('get_existing_model() can only be used in interactive mode.')
            return None

    def add_options_py(self, model, save_dir, metric, reduce_lr_on_plateau=None, save_checkpoints=None, early_stopping=None):
        '''
        If in interactive mode, asks the user whether to add learning rate reduction, checkpoint saving and early stopping.
        If a config file is used the options are set based on the provided paramaters. In interactive mode default values
        are used, whereas they can be changed if a config file is used.
        '''
        if self.backend == self.PY_BACKEND:
            if self.interactive_mode:
                msg = "Save checkpoints? (y/n)"
                val = self.yes_no_question(msg)
                if val in [self.QUIT, self.BACK]: return val

                if val == 'y': 
                    print('Saving checkpoints to {}'.format(save_dir))
                    model.add_checkpoint_saver(metric,save_dir)
            elif save_checkpoints: model.add_checkpoint_saver(metric,save_dir)
            
            if self.interactive_mode:
                val = self.yes_no_question(self.LR_DECR)
                if val in [self.QUIT, self.BACK]: return val
                
                if val == 'y': 
                    model.reduce_lr_on_plateau(metric_name=metric, factor=0.7, patience=5)
            elif reduce_lr_on_plateau is not None:
                factor =  reduce_lr_on_plateau['factor']
                patience =  reduce_lr_on_plateau['patience']
                self.logger.info('Learning rate reduction: factor: {} patience: {}'.format(factor,patience))
                model.reduce_lr_on_plateau(metric_name=metric, factor=factor, patience=patience)

            if self.interactive_mode:
                val = self.yes_no_question('Early stopping? (y/n)')
                if val in [self.QUIT, self.BACK]: return val
                
                if val == 'y': 
                    model.add_early_stopping(metric_name=metric, patience=20)
            elif early_stopping is not None: 
                self.logger.info('Early stopping with patience {} epochs'.format(early_stopping))
                model.add_early_stopping(metric_name=metric, patience=early_stopping)

            return model
        else: 
            self.logger.warning('Options can be added only with PyTorch backend.')
            return None

    def load_from_model_dir_maybe(self):

        msg = "Would you like to set the model directory?\n(If a model is present, it will be reused.) (y/n)"
        val = self.yes_no_question(msg)

        if val == 'y': 
            print("Specify the path to the model.")
            path = input('>>> ')
            while not os.path.exists(path):
                print('The path does not exist.')
                path = input('>>> ')
                if path in [self.BACK, self.QUIT]: 
                    return path
            return path
        return self.BACK

    def build_dir_structure(self, initial_dir, final_dir_name=None, params=None, overwrite_model_dir=None):
        path_first_part = os.path.join(initial_dir,self.date_string)

        final_dir = ''
        if self.interactive_mode:
            if self.backend == self.TF_BACKEND: path = self.load_from_model_dir_maybe()
            else: path = self.BACK

            if path not in [self.QUIT, self.BACK]: final_dir = path
            else: 
                temp_path = os.path.join(path_first_part,'interactive')
                if final_dir_name:
                    final_dir = self.add_path_suffix_maybe(temp_path, final_dir_name)
                else: final_dir = temp_path
        else:
            temp_path = os.path.join(path_first_part,'grid_search',self.cur_config_command['model_dir'])
            if final_dir_name: final_dir = self.add_path_suffix_maybe(temp_path, final_dir_name)
            else: final_dir = temp_path

        final_dir = final_dir.replace('[','')
        final_dir = final_dir.replace(']','')
        final_dir = final_dir.replace(', ','_')

        util.makedir_maybe(final_dir)
        if params is not None:
            with open(os.path.join(final_dir,'params.json'), 'w') as f:
                json.dump(params, f, indent=4, sort_keys=True)

        return final_dir

    def build_cl_tf(self, params, weights=None, biases=None, force_new_dir=None):

        cl_dir = self.build_dir_structure(self.CHECKPOINT_PATH_TF, 'cl', params=params)
        self.logger.info('Storing TensorFlow classifier in {}'.format(cl_dir))
        
        hn = params['hn']
        if not isinstance(hn,list): hn = [hn]
        if hn[0] == 0: hn = []

        if len(hn) > 0: af = util.extend_param_list(params['af'], len(hn))
        else: af = []

        if weights is not None and not isinstance(weights,list): weights = [weights]
        if biases is not None and not isinstance(biases,list): biases = [biases]

        classifier = TF_Classifier(
            feature_columns=self.dataset['feature_cols'], 
            hidden_layers=hn, num_classes=self.dataset['n_classes'], 
            weight_decay=params['wd'], train_last_only=params.get('last_only'),
            weights=weights, biases=biases,dropout=params.get('do'),
            batch_norm=params.get('bn'), activation_fns=af,
            learning_rate=params['lr'], model_dir=cl_dir,
        )

        self.logger.info(classifier)

        return classifier

    def build_cl_py(self, params, weights=None, biases=None, force_new_dir=None):
        tensorboard_dir = self.build_dir_structure(self.SUMMARY_DIR_PY, 'cl', params=params)
        checkpoint_dir = self.build_dir_structure(self.CHECKPOINT_PATH_PY, 'cl', params=params)

        self.logger.info('Storing PyTorch classifier in {}'.format(tensorboard_dir))

        in_size = self.dataset['train'].input_size()
        out_size = self.dataset['n_classes']

        af = params['af']
        af = af if isinstance(af,list) else [af]

        hn = params['hn']
        if not isinstance(hn,list): hn = [hn]
        if hn[0] == 0: hn = None

        if hn is not None: af = util.extend_param_list(af, len(hn))
            
        if weights is not None and not isinstance(weights,list): weights = [weights]
        if biases is not None and not isinstance(biases,list): biases = [biases]
        
        cl = PY_Classifier(
            input_size=in_size, hidden_layers=hn, 
            activation_fns=af, num_classes=out_size,  
            weights=weights, biases=biases, train_last_only=params.get('last_only'),
            dropout=params.get('do'),batch_norm=params.get('bn'), 
            tensorboard_dir=tensorboard_dir,
            logger=self.logger
        )

        cl = self.add_options_py(cl,checkpoint_dir,metric='accuracy',
            reduce_lr_on_plateau=params.get('reduce_lr'),
            save_checkpoints=params.get('cs'), 
            early_stopping=params.get('es')
        )

        self.logger.info(cl)

        return cl

    def build_ae_py(self, params, force_new_dir=None):
        
        tensorboard_dir = self.build_dir_structure(self.SUMMARY_DIR_PY, 'ae', params=params)
        checkpoint_dir = self.build_dir_structure(self.CHECKPOINT_PATH_PY, 'ae', params=params)
        self.logger.info('Storing TensorBoard data in {}'.format(tensorboard_dir))

        in_size = self.dataset['train'].input_size()

        ae = PY_Autoencoder(
            input_size=in_size, num_hidden_neurons=params['hn'], 
            kl_weight=params['kl'], activation_fn=params['af'], 
            dropout=params.get('do'),batch_norm=params.get('bn'), 
            rho=params['rho'], logger=self.logger, 
            tensorboard_dir=tensorboard_dir
        )
        
        ae = self.add_options_py(
            ae,checkpoint_dir,metric='kl',
            reduce_lr_on_plateau=params.get('reduce_lr'),
            save_checkpoints=params.get('cs'), 
            early_stopping=params.get('es')
        )

        return ae


    def build_ae_tf(self, params, force_new_dir=None):

        ae_dir = self.build_dir_structure(self.CHECKPOINT_PATH_TF, 'ae', params=params)
        self.logger.info('Storing TensorFlow autoencoder in {}'.format(ae_dir))

        ae = TF_Autoencoder(
            feature_columns=self.dataset['feature_cols'], kl_weight=params['kl'], 
            weight_decay=params['wd'], num_hidden_neurons=params['hn'], dropout=params.get('do'),
            batch_norm=params.get('bn'), learning_rate=params['lr'], model_dir=ae_dir
        )
        
        self.logger.info(ae)

        return ae

    def build_sae_py(self, params, force_new_dir=None):

        tensorboard_dir = self.build_dir_structure(self.SUMMARY_DIR_PY, 'sae', params=params)
        checkpoint_dir = self.build_dir_structure(self.CHECKPOINT_PATH_PY, 'sae', params=params)

        in_size = self.dataset['train'].input_size()

        hn = params['hn']
        hn = hn if isinstance(hn,list) else [hn]

        af = util.extend_param_list(params['af'], len(hn))
        rho = util.extend_param_list(params['rho'], len(hn))
        kl = util.extend_param_list(params['kl'], len(hn))

        sae = PY_StackedAutoencoder(
            input_size=in_size, hidden_layers=hn, 
            kl_weight=kl, activation_fns=af,
            dropout=params.get('do'),batch_norm=params.get('bn'), 
            rho=rho, logger=self.logger, tensorboard_dir=tensorboard_dir
        )

        sae = self.add_options_py(
            sae,checkpoint_dir,metric='kl',
            reduce_lr_on_plateau=params.get('reduce_lr'),
            save_checkpoints=params.get('cs'), 
            early_stopping=params.get('es')
        )

        self.logger.info(sae)

        return sae

    def build_sae_tf(self, params: Dict[str,Any], force_new_dir=None):
        
        sae_dir = self.build_dir_structure(self.CHECKPOINT_PATH_TF, 'sae', params=params)
        self.logger.info('Storing TensorFlow stacked autoencoder in {}'.format(sae_dir))

        hn = params['hn']
        hn = hn if isinstance(hn,list) else [hn]

        af = util.extend_param_list(params['af'], len(hn))
        wd = util.extend_param_list(params['wd'], len(hn))
        rho = util.extend_param_list(params['rho'], len(hn))
        kl = util.extend_param_list(params['kl'], len(hn))

        sae = TF_StackedAutoencoder(
            feature_columns=self.dataset['feature_cols'],
            hidden_layers=hn, weight_decay=wd,
            kl_weight=kl, rho=rho, dropout=params['do'],
            activation_fns=af,
            batch_norm=params['bn'], model_dir=sae_dir
        )

        self.logger.info(sae)

        return sae

    def train_sae_py(self, s_autoencoder, params):
        train_ds, test_ds = self.dataset['train'], self.dataset['test']
        
        if isinstance(params['hn'], int): n_hidden = 1
        else: n_hidden = len(params['hn'])
        ne = util.extend_param_list(params['ne'], n_hidden)
        wd = util.extend_param_list(params['wd'], n_hidden)
        lr = util.extend_param_list(params['lr'], n_hidden)

        s_autoencoder.train(
            train_ds=train_ds(params['bs'], shuffle=True), 
            test_ds=test_ds(params['bs'], shuffle=True), 
            list_epochs=ne, weight_decay=wd, 
            learning_rates=lr, metrics=['kl','mse'],
            log_every_n=500
        )
        return s_autoencoder

    def train_sae_tf(self, s_autoencoder, params):
        train_ds, test_ds = self.dataset['train'], self.dataset['test']
        hn = params['hn']
        hn = hn if isinstance(hn,list) else [hn]

        ne = util.extend_param_list(params['ne'], len(hn))
        lr = util.extend_param_list(params['lr'], len(hn))

        s_autoencoder.train(
            train_ds=train_ds['features'], 
            batch_size=params['bs'], epochs=ne, learning_rates=lr, 
            test_ds=test_ds['features']
        )

        return s_autoencoder

    def train_ae_py(self, autoencoder, params): 
        train_ds, test_ds = self.dataset['train'], self.dataset['test']

        autoencoder.train(
            train_ds=train_ds(params['bs'], shuffle=True), 
            test_ds=test_ds(params['bs'], shuffle=False), 
            weight_decay=params['wd'], learning_rate=params['lr'], 
            num_epochs=params['ne'], metrics=['kl','mse'], log_every_n=500
        )
        return autoencoder

    def train_ae_tf(self, autoencoder, params):
        train_ds, test_ds = self.dataset['train'], self.dataset['test']

        for _ in range(params['ne']):
            autoencoder.train(train_ds['features'], params['bs'], 1)
            autoencoder.evaluate(test_ds['features'])

        return autoencoder

    def train_cl_py(self, classifier, params, cross_val_dataset=None): 
        if cross_val_dataset is None:
            train_ds, test_ds = self.dataset['train'], self.dataset['test']
        else: 
            train_ds, test_ds = cross_val_dataset['train'], cross_val_dataset['test']

        classifier.train(
            train_ds=train_ds(params['bs'], shuffle=True), 
            test_ds=test_ds(params['bs'], shuffle=False), 
            weight_decay=params['wd'], learning_rate=params['lr'], 
            num_epochs=params['ne'], log_every_n=500,
            metrics=['accuracy', 'precision', 'recall', 'f_measure']
        )
        return classifier

    def train_cl_tf(self, classifier, params):
        train_ds, test_ds = self.dataset['train'], self.dataset['test']

        for _ in range(params['ne']):
            classifier.train(train_ds['features'], params['bs'], 1, train_labels=train_ds['labels'])
            classifier.evaluate(test_ds['features'], test_labels=test_ds['labels'])

        return classifier

    def parserV2(self, params, model, options=False):
        hyperparams = {}
        for key in params: 
            if options: 
                val = self.converters[key][model](params[key])
            else: val = [self.converters[key][model](val) for val in params[key]]
            if (options and val is None) or \
               (not options and any([x is None for x in val])):
                raise ValueError('Error while parsing \'{}\':{} in {}'.format(key,params[key],params))
            hyperparams[key] = val
        return hyperparams

    def programm_loop(self):

        command = None
        internal_command = None
        redirect = None

        if self.interactive_mode:
            val = self.question_loop(self.BACKEND_MSG, allowed=[self.TF_BACKEND,self.PY_BACKEND], input_text='>>> ')
            if val == self.QUIT: return
            self.backend = val
            if self.backend == self.PY_BACKEND: self.logger = util.setup_logger(self.LOGGING_PATH_PY)
            else: self.logger = util.setup_logger(self.LOGGING_PATH_TF)
        else: internal_command = self.pop_command()
    
        while True: 
            if command == self.QUIT: 
                if not self.interactive_mode: break
                if self.yes_no_question(self.QUIT_PROMPT) == 'y': break

            while True:
                if internal_command: 
                    command = internal_command
                    internal_command = None
                else: command = input('>>> ').strip()

                if command == self.LOAD: 
                    if not self.interactive_mode: print('Dataset loaded. Skipping...')
                    else:
                        val = self.question_loop(self.LOAD_DS_MSG, allowed=[1,2,3], convert_fn=util.convert_to_int, input_text='>>> ')
                        if val in [self.QUIT, self.BACK]: 
                            command = val
                            break

                        dataset_num = val

                        start = datetime.now()
                        self.dataset = self.fns[command][self.backend](dataset_num)
                        end = datetime.now()
                        print(self.DATASET_LOAD_FIN_MSG.format((end-start).total_seconds()))

                        if redirect is not None: 
                            internal_command = redirect
                            redirect = None

                elif command == self.AE:
                    ae_index = -1

                    if not self.interactive_mode:
                        print(self.COMMAND_ONLY_INTERACTIVE.format(command))
                        if len(self.config['commands']) == 0:
                            command = self.QUIT
                            break

                        internal_command = self.pop_command()

                    elif self.dataset: 
                        val = self.use_existing_model_maybe(self.autoencoders)
                        if val in [self.QUIT, self.BACK]: 
                            command = val
                            break
                        ae_index = val
                        use_existing = ae_index != -1

                        print(self.CONFIG_MSG.format('autoencoder'))
                        if use_existing: 
                            ae = self.autoencoders[ae_index].model
                            param_dict = self.autoencoders[ae_index].param_dict
                            if self.backend == self.PY_BACKEND:
                                config = self.config_ae_short
                            else: config = self.config_ae_short_tf
                            val = self.config_loop(config, param_dict)
                            if val in [self.QUIT,self.BACK]: 
                                command = val
                                break

                            self.params_ae = val
                        else: 
                            val = self.config_loop(self.config_ae, self.params_ae)
                            if val in [self.QUIT,self.BACK]: 
                                command = val
                                break
                                
                            self.params_ae = val
                            
                            ae = self.get_build_fn(command)(self.params_ae)
                            self.autoencoders.append(self.ModelCache(ae, deepcopy(self.params_ae)))
                        
                        ae = self.get_train_fn(command)(ae, self.params_ae)

                        if redirect is not None: 
                            internal_command = redirect
                            redirect = None
                            
                    else: 
                        internal_command = self.LOAD
                        redirect = self.AE
                    
                elif command == self.CL:

                    if not self.interactive_mode:
                        print(self.COMMAND_ONLY_INTERACTIVE.format(command))
                        if len(self.config['commands']) == 0:
                            command = self.QUIT
                            break

                        internal_command = self.pop_command()

                    elif self.dataset:
                        weights, biases = None, None
                        use_ae = False
                        if self.available_aes():
                            combined = self.autoencoders + self.s_autoencoders
                            val = self.use_existing_model_maybe(combined, prompt=self.AE_PARAMS)
                            if val in [self.QUIT, self.BACK]:
                                command = val
                                break

                            ae_idx = val
                            
                            use_ae = ae_idx != -1

                        build_classifier = True
                        if use_ae:
                            ae = combined[ae_idx].model
                            weights, biases = ae.get_weights_and_biases()
                            if not isinstance(weights,list): weights = [weights]
                            if not isinstance(biases,list): biases = [biases]
                            print('Getting weights and biases...')
                            self.params_cl['hn'] = ae.neurons_per_layer
                            self.params_cl['af'] = ae.activation_fn_names
                            print(self.CONFIG_MSG.format('classifier'))
                            val = self.config_loop(self.config_cl_from_ae, self.params_cl)
                            if val in [self.QUIT,self.BACK]: 
                                command = val
                                break

                            params_cl = val

                        else:
                            val = self.use_existing_model_maybe(self.classifiers, prompt=self.CL_PROMPT)
                            if val in [self.QUIT, self.BACK]:
                                command = val
                                break

                            cl_idx = val
                            use_existing = cl_idx != -1

                            if use_existing:
                                cl = self.classifiers[cl_idx].model
                                param_dict = self.classifiers[cl_idx].param_dict
                                if self.backend == self.PY_BACKEND:
                                    config = self.config_cl_short
                                else: config = self.config_cl_short_tf

                                print(self.CONFIG_MSG.format('classifier'))
                                val = self.config_loop(config, param_dict)
                                if val in [self.QUIT,self.BACK]: 
                                    command = val
                                    break
                                params_cl = val
                                build_classifier = False

                            else:
                                print(self.CONFIG_MSG.format('classifier'))
                                val = self.config_loop(self.config_cl, self.params_cl)
                                if val in [self.QUIT,self.BACK]: 
                                    command = val
                                    break
                                params_cl = val

                        if build_classifier:
                            cl = self.get_build_fn(command)(params_cl, weights, biases)
                            self.classifiers.append(self.ModelCache(cl, deepcopy(params_cl)))
                        
                        cl = self.get_train_fn(command)(cl, params_cl)
                            
                        if redirect is not None: 
                            internal_command = redirect
                            redirect = None
                    else: 
                        internal_command = self.LOAD
                        redirect = self.CL

                elif command == self.SAE: 
                    if not self.interactive_mode:
                        print(self.COMMAND_ONLY_INTERACTIVE.format(command))
                        if len(self.config['commands']) == 0:
                            command = self.QUIT
                            break

                        internal_command = self.pop_command()

                    elif self.dataset: 
                        sae_index = -1
                        val = self.use_existing_model_maybe(self.s_autoencoders)
                        if val in [self.QUIT, self.BACK]: 
                            command = val
                            break

                        sae_index = val
                        use_existing = sae_index != -1

                        if use_existing: 
                            sae = self.s_autoencoders[sae_index].model
                            param_dict = self.s_autoencoders[sae_index].param_dict
                            if self.backend == self.PY_BACKEND: config = self.config_sae_short
                            else: config = self.config_sae_short_tf

                            print(self.CONFIG_MSG.format('stacked autoencoder'))
                            val = self.config_loop(config, param_dict)
                            if val in [self.QUIT,self.BACK]: 
                                command = val
                                break

                            params_sae = param_dict
                        else: 
                            print(self.CONFIG_MSG.format('stacked autoencoder'))
                            val = self.config_loop(self.config_sae, self.params_sae)
                            if val in [self.QUIT,self.BACK]: 
                                command = val
                                break
                                
                            params_sae = val
                            
                            sae = self.get_build_fn(command)(params_sae)
                            self.s_autoencoders.append(self.ModelCache(sae, deepcopy(params_sae)))

                        sae = self.get_train_fn(command)(sae, params_sae)  
                    else: 
                        internal_command = self.LOAD
                        redirect = self.SAE
                    

                elif command == self.GRID_SEARCH:
                    if self.interactive_mode: 
                        print('Grid search can only be performed via a JSON configuration file. Run with --config <CONFIG PATH>')
                        command = self.BACK
                        break

                    # Switch dataset if specified in command
                    if 'dataset' in self.cur_config_command:
                        self.logger.info('Changing dataset...')
                        pca = None
                        dataset = self.cur_config_command['dataset']
                        if isinstance(dataset,dict): 
                            dataset_name = dataset['name']
                            pca = dataset.get('pca')
                        else: dataset_name = dataset

                        self.dataset = self.load_dataset(self.dataset_name_to_selector(dataset_name), pca=pca)

                    if 'model_dir' in self.cur_config_command:
                        if self.backend == self.PY_BACKEND:
                            checkpoint_dir = self.CHECKPOINT_PATH_PY
                        else: checkpoint_dir = self.CHECKPOINT_PATH_TF 

                        model_dir = self.cur_config_command['model_dir']
                        temp_dir = os.path.join(checkpoint_dir, self.date_string, 'grid_search')
                    else: model_dir = 'gs'

                    self.cur_config_command['model_dir'] = self.get_dir_with_suffix(temp_dir, model_dir)

                    # Switch logging
                    path = self.LOGGING_PATH_PY if self.backend == self.PY_BACKEND else self.LOGGING_PATH_TF
                    path = self.build_dir_structure(path)
                    self.logger = util.setup_logger(path, name_suffix=self.cur_config_command['model_dir'])

                    parsed = self.cur_config_command['parsed']
                    model_1, model_2 = parsed['model_1']['name'], parsed['model_2']['name']
                    m1_merged, m2_merged = parsed['model_1']['params'], parsed['model_2']['params']
                    m1_options, m2_options = parsed['model_1']['options'], parsed['model_2']['options']

                    # The case when an autoencoder  
                    # will be used for pretraining
                    if model_2 is not None: 
                        final_results = util.grid_search(
                            self.get_grid_search_train_fn(
                                main_model=model_1, options=m1_options, classifier=model_2, 
                                classifier_params=m2_merged, classifier_options=m2_options
                            ), m1_merged
                        )
                    else: final_results = util.grid_search(self.get_grid_search_train_fn(main_model=model_1, options=m1_options), m1_merged)

                    # When PyTorch is used a file results_*.pickle is saved
                    # which holds the best evaluation results for each run
                    if self.backend == self.PY_BACKEND:
                        path = self.build_dir_structure(self.CHECKPOINT_PATH_PY, 'results')

                        with open(os.path.join(path,'results.json'), 'w') as f:
                            json.dump(final_results, f, indent=4, sort_keys=True)

                        self.logger.info(final_results)

                    self.num_grid_searches += 1

                    internal_command = self.pop_command()
                
                elif command == self.HELP: print(self.GENERAL_HELP)
                elif command == self.QUIT: break
                elif command in ['', self.BACK]: pass
                else: print(self.UNKNOWN_COMMAND.format(command))
