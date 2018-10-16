
import warnings 
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import pandas as pd
    import tensorflow as tf
    import numpy as np
    import logging
import sys
import os
from typing import List, Callable
from datetime import datetime

def create_text_file(path, name, content):
    full = os.path.join(path,name + '.txt')
    with open(full, 'w') as f:
        f.write(content)

def get_date_string():
    now = datetime.now() 
    return str(now.day) + '_' + str(now.month) + '_' + str(now.year)

def convert_to_int(value, min_value=None, max_value=None):
    return_val = None
    has_fraction = False
    try: value.index('.')
    except: pass
    else: has_fraction = True

    if not has_fraction:
        try: return_val = int(value)
        except ValueError: pass

    if return_val is not None and \
       (min_value is not None and return_val <= min_value) or \
       (max_value is not None and return_val >= max_value):
        return None

    return return_val

def convert_to_float(value, min_value=None, max_value=None):
    return_val = None
    try: return_val = float(value)
    except ValueError: pass

    if return_val is not None and \
       ((min_value is not None and return_val <= min_value) or \
       (max_value is not None and return_val >= max_value)):
        return None

    return return_val

def overwrite_dict(defaults, new_vals):
    overwritten = {}
    for key in defaults: 
        if key in new_vals: overwritten[key] = new_vals[key]
        else: overwritten[key] = defaults[key]
    return overwritten
                    
def dict_entries_to_list(dict_to_convert):
    new_dict = {}
    for key in dict_to_convert: 
        if not isinstance(dict_to_convert[key],list):
            new_dict[key] = [dict_to_convert[key]]
        else: new_dict[key] = dict_to_convert[key]

    return new_dict

def convert_to_list(data: str, map_fn: Callable=None):
    '''
        Converts a string of comma-separated items into a list. 
        Ignores empty elements and white-space.

        Args: 
          data - the string to be converted to list
          map_fn - An optional function that takes each value and performs some transformation. None can be 
          returned for values that do not meet some condition.

        Returns: 
          The converted list. If None values are returned from map_fn, None is returned.
    '''
    as_list = data.split(',')
    as_list = [elem.strip() for elem in as_list]
    as_list = [elem for elem in as_list if elem != '']

    if map_fn: as_list = [map_fn(elem) for elem in as_list]
    if None in as_list: return None

    return as_list

def extend_param_list(param, length, param_name=None):
        '''
        Check if the parameter is a list or a single value. In the first case
        an error is raised if the size of the list is bigger than one and does not correspond to the 
        number of layers. If a single value or a one-element list is given, the value is repeated for each layer.
        '''
        if isinstance(param, list):
            if len(param) < length:
                if len(param) == 1: return param * length
                else: return [param[0]] * length
                    # if param_name:
                    #     raise ValueError('Parameter "{}" contains {} elements but {} are required.'.format(param_name, len(param), length))
                    # else: 
                    #     raise ValueError('The parameter contains {} elements but {} are required.'.format(len(param), length))
            elif len(param) > length: return param[:length]
            else: return param
        return [param] * length


def makedir_maybe(directory):
    '''Creats a the specified directory if it
    doesn't exist. Returns True if a directory
    was created, False otherwise.'''
    created = False
    if not os.path.exists(directory):
        os.makedirs(directory)
        created = True
    return created

def get_config(*params, name_val_bind=None, tuple_bind=None, verbose=True):
    '''Builds a string out of the provided values in *params, which can
    be used for logging purposes or saving data. With other parameters set to None,
    the string has the form: 

      'Configuration: metric_name: metric_value; ...

      Args: 
        *params - tuples of the form (name, value), where name is the
        name of the metric and value  holds its respective value.
        name_val_bind - string used to connect the name and the value of the
        metric. Default is ": ".
        tuple_bind - string used to bind two tuples. Default is "; "
        verbose - whether to include "Configuration: " at the start.

    '''

    if not name_val_bind: name_val_bind = ': '
    if not tuple_bind: tuple_bind = '; '
    msg = ''
    if verbose: msg = "Configuration: "
    param_list = []
    for name, value in params: 
        if value is not None:
            msg += '{}'+ name_val_bind + '{}' + tuple_bind
            param_list.extend([name,value])
    
    msg = msg[:-len(tuple_bind)]
    return msg.format(*param_list)


def grid_search(train_fn, hyperparams, other_params=None):
    '''
      Args: 
        train_fn: A function that takes a number, indicating the current iteration as its first argument and the same dictionary as the one given to hyperparams, but with single values instead of lists as the corresponding value. The function can optionally return a dictionary of the form {metric_name:[...], ..., hyperparam_name:[...], ...}, with same 
        list indices holding information about the hyperparameters used for the different runs and the results obtained. The dictionary is updated at each run.
        hyperparams: a dictionary with each keyword representing a hyperparameter and holding a list with values to be tested.

      Returns: 
        The dictionary returned by train_fn if train_fn is set to return one.
    '''
    
    return grid_search_internal(hyperparams, train_fn, other_params=other_params)

def grid_search_internal(param_dict, train_fn, length=None, iteration_list=None, set_args=None, other_params=None, final_results=None):
    if length is None: length = len(param_dict)
    if iteration_list is None: iteration_list = []
    if set_args is None: set_args = {}
        
    # if all parameters have been gathered
    if len(set_args) == length: 
        iteration_list.append('')
        val = train_fn(len(iteration_list), set_args, other_params)
        if val is not None: 
            final_results = val
            return final_results
        return
        
    # Take a key and iterate over the possible parameters
    # Call recursively with all keys excluding the selected one,
    # which is added to set_args, holding a single value
    some_key = next(iter(param_dict)) 
    for item in param_dict[some_key]:
        set_args[some_key] = item
        new_dict = {k: v for k, v in param_dict.items() if k != some_key}
        val = grid_search_internal(new_dict, train_fn, length, iteration_list, set_args=set_args.copy(),other_params=other_params, final_results=final_results)
        if val is not None: final_results = val

    return final_results

def merge_dicts(dict_1, dict_2):
    '''
        Merges the values of the two dictionaries. Values are concatenated into a list.
        The two dictionaries must either have the same set of keys or one of them
        has to be empty. Otherwise a ValueError is thrown. 
    '''

    keys_1 = set(dict_1.keys())
    keys_2 = set(dict_2.keys())
    union = keys_1.union(keys_2)
    
    if len(keys_1) > 0 and len(keys_2) > 0 and (len(keys_1) < len(union) or len(keys_2) < len(union)):
        raise ValueError('Non-empty dictionaries with differing sets of keys.')
        
    if len(keys_1) == 0:
        return dict_2
        
    if len(keys_2) == 0:
        return dict_1
    
    new_dict = {}
    for key in dict_1:
        print(key)
        if not isinstance(dict_1[key],list):
            dict_1[key] = [dict_1[key]]
            
        if not isinstance(dict_2[key],list):
            dict_2[key] = [dict_2[key]]
            
        new_dict[key] = dict_1[key] + dict_2[key]
    
    return new_dict


# def k_fold_cross_val(dataset, train_fn, val_fn, folds=10, shuffle=True):
#     '''
#         Performs cross validation on the provided dataset.

#         Args:
#             dataset (DataFrame) - the dataset on which to perform cross-validation as Pandas DataFrame 
#             train_fn (fuction) - a training function that uses the training split of the dataset
#             val_fn (fuction) - a validation function that uses the validation split of the dataset
#             folds - number of cross-validation folds
#             shuffle (Bool) - whether dataset should be shuffled beforehand

#         Both train_fn and val_fn can return a dict of values of interest e.g. (loss, accuracy, etc.)

#         Returns:
#             A tuple of traing and validation dict with same keys as returned from train_fn and val_fn.
#             For each key a list of the values over the trining folds is provided

#     '''
#     ds = dataset
#     if folds > len(ds):
#         raise ValueError('More folds than elements. {} folds, but only {} elements.'.format(folds, len(ds)))
    
#     if shuffle:
#         ds = ds.sample(frac=1).reset_index(drop=True)
    
#     res_dict_train = {}
#     res_dict_val = {}

#     per_fold = len(ds) // folds
#     for i in range(folds):
#         train_fold = pd.concat([ds.iloc[:i*per_fold], ds.iloc[i*per_fold + per_fold:]])
#         test_fold = ds.iloc[i*per_fold:i*per_fold + per_fold]
        
#         new_dict_train = train_fn(train_fold)
#         new_dict_val = val_fin(test_fold)

#         res_dict_train = merge_dicts(res_dict_train,new_dict_train)
#         res_dict_val = merge_dicts(res_dict_val,new_dict_val)
    
#     return res_dict_train, res_dict_val

def get_uniques(*series):
    '''
        For the provided Pandas Series objects, a sorted list is returned that holds all unique values found in them by performing union operation.
    '''
    if len(set([s.dtype for s in series])) > 1: 
        raise ValueError('All series must have the same data type.')
        
    uniques = set()
    for serie in series: 
        uniques = uniques.union(set(serie))
    return sorted(uniques)

def get_value_to_num_dict(possible_values, sort=False):
    '''
        Returns a dictionary with each value in possible_values being a key that returns an integer.
    '''
    if sort: possible_values = sorted(possible_values)
    indices = [i for i in range(len(possible_values))]
    return {name: value for name, value in zip(possible_values, indices)}
    

def string_to_numeric(*pd_series, transformation_dict):
    '''
        Args: 
           *pd_series - Pandas Series objects onto which the transformation will be applied
           transformation_dict - A dict where each key is found in the provided Series objects and the returned value is an integer
        Returns:
            The provided Series objects with the applied transformation
    '''
    return_series = []
    for series in pd_series: 
        return_series.append(series.apply(lambda cur: transformation_dict[cur]))
    return tuple(return_series)

def one_hot_vector(idx, num_unique, dtype=None):
    one_hot_vector = np.zeros(num_unique)
    if dtype: one_hot_vector = one_hot_vector.astype(dtype)
    one_hot_vector[idx] = 1

    return one_hot_vector

def min_max_normalize(dataframe, column_headers):
    cur = 0
    for header in column_headers:
        cur += 1
        print("Normalizing {} of {}".format(cur, len(column_headers)))
        dataframe[header] = min_max_normalize_series(dataframe[header], dataframe[header].min(), dataframe[header].max())
    return dataframe

def min_max_normalize_series(series, min, max):
    return series.apply(lambda elem: (elem - min) / (max - min))

def get_normalization_headers(dataframe):
    numeric_df = dataframe.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']) 
    normalization_headers = []
    for series in numeric_df:
        if series == 'attack_difficulty':
            continue

        if int(numeric_df[series].max()) > 1 or int(numeric_df[series].min()) < 0:
            normalization_headers.append(series)
    return normalization_headers

def setup_logger(path=None, name_suffix=None, only_console=False, no_console=False):
    debug_path = 'debug.log'
    info_path = 'info.log'

    if name_suffix:
        debug_path = 'debug_' + name_suffix + '.log'
        info_path = 'info_' + name_suffix + '.log'

    if path:
        debug_path = os.path.join(path, debug_path)
        info_path = os.path.join(path, info_path)

    if name_suffix: name = name_suffix
    else: name = __name__

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    debug_logger = logging.FileHandler(debug_path, mode='w')
    debug_logger.setLevel(logging.DEBUG)
    debug_logger.setFormatter(formatter)

    info_logger = logging.FileHandler(info_path, mode='w')
    info_logger.setLevel(logging.INFO)
    info_logger.setFormatter(formatter)

    console_logger = logging.StreamHandler(sys.stdout)
    console_logger.setLevel(logging.DEBUG)
    console_logger.setFormatter(formatter)

    if only_console is False:
        logger.addHandler(debug_logger)
        logger.addHandler(info_logger)

    if no_console is False:
        logger.addHandler(console_logger)

    return logger

# TODO: Probably remove
# Formats the logging of the provided tensors 
def custom_logging(tensor_values, tags):
    for value in tags:
        tf.logging.info("%s = \n%s" % (value, tensor_values[value]))  



