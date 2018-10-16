import pandas as pd
import tensorflow as tf

import os

def build_kdd_dataset(train_path, test_path):
    '''
     Args: 
        train_path - Path to a CSV file that holds NSL-KDD training data 
        test_path - Path to the CSV file that holds NSL-KDD test data

    Returns: 
        A tuple that contains two dicts and the feature columns. First dict holds training data and the second - test data. Each dict has keys 'features' and 'labels'.
    '''
    label_header = 'attack'

    train_features, train_labels = get_kdd_features_and_labels(train_path)
    test_features, test_labels = get_kdd_features_and_labels(test_path)

    unique_labels = get_uniques(train_labels[label_header],test_labels[label_header])
    attack_to_int_dict = get_label_dict(unique_labels)
    train_labels[label_header], test_labels[label_header] = string_to_numeric(train_labels[label_header], test_labels[label_header], transformation_dict=attack_to_int_dict)
    feature_columns = build_feature_columns(train_features)

    return {'features':train_features, 'labels':train_labels}, {'features':test_features, 'labels':test_labels}, feature_columns

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


def get_kdd_features_and_labels(file_location, small_test=False):
    '''
    Splits NSL-KDD dataset into features and labels. Returns a tuple with the two Pandas DataFrame objects.
    '''
    dataset = pd.read_csv(file_location, sep=",")
    dataset = dataset.drop(columns=['41_num_outbound_cmds'])
    return (dataset.iloc[:, 0:4], dataset.iloc[:, -2:-1]) if small_test else (dataset.iloc[:, 0:-2], dataset.iloc[:, -2:-1])


def get_label_dict(possible_values):
    '''
        Returns a dictionary with each value in possible_values being a key that returns an integer.
    '''
    indices = [i for i in range(len(possible_values))]
    return {name: value for name, value in zip(possible_values, indices)}


def build_feature_columns(dataset):
    '''
        Generates numeric feature columns for continious features and categorical columns for features represented by a string.     

        Args: 
            dataset - a Pandas DataFrame object that contains only features and no labels

        Returns: 
            A list of feature columns
    '''

    # Returns a list of the column names
    headers = sorted(list(dataset)) 

    feature_columns = []
    for header in headers: 

        # String data has the type 'object'
        if dataset[header].dtype == 'object':

            # Sorting is important in vocabulary_list, since  
            # otherwise different builds will have different 
            # one-hot encodings, which is not ideal for debugging.
            categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key=header,
                vocabulary_list=sorted(set(dataset[header]))
            )
            feature_columns.append(tf.feature_column.indicator_column(categorical_column))
        else:
            feature_columns.append(tf.feature_column.numeric_column(key=header))

    return feature_columns
    

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


# Formats the logging of the provided tensors 
def custom_logging(tensor_values, tags):
    for value in tags:
        tf.logging.info("%s = \n%s" % (value, tensor_values[value]))  



