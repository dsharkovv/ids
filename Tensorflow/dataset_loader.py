import pandas as pd 
import tensorflow as tf
import util

FULL_TRAIN_PATH = 'Dataset/Train/KDDTrain.csv'
FULL_TEST_PATH = 'Dataset/Test/KDDTest.csv' 

TWOC_TRAIN_PATH = 'Dataset/Train/KDDTrain_2c.csv'
TWOC_TEST_PATH = 'Dataset/Test/KDDTest_2c.csv' 

FIVEC_TRAIN_PATH = 'Dataset/Train/KDDTrain_5c.csv'
FIVEC_TEST_PATH = 'Dataset/Test/KDDTest_5c.csv' 

def build_kdd_dataset(train_path, test_path):
    '''
     Args: 
        train_path - Path to a CSV file that holds NSL-KDD training data 
        test_path - Path to the CSV file that holds NSL-KDD test data

    Returns: 
        A tuple that contains two dicts, the feature columns and the number of classes. First dict holds training data and the second - test data. Each dict has keys 'features' and 'labels'.
    '''
    label_header = 'attack'

    train_features, train_labels = get_kdd_features_and_labels(train_path)
    test_features, test_labels = get_kdd_features_and_labels(test_path)

    unique_labels = util.get_uniques(train_labels[label_header],test_labels[label_header])
    attack_to_int_dict = util.get_value_to_num_dict(unique_labels)
    train_labels[label_header], test_labels[label_header] = util.string_to_numeric(train_labels[label_header], test_labels[label_header], transformation_dict=attack_to_int_dict)
    feature_columns = build_feature_columns(train_features)

    return {'features':train_features, 'labels':train_labels}, {'features':test_features, 'labels':test_labels}, feature_columns, len(unique_labels)


def get_kdd_features_and_labels(file_location, small_test=False):
    '''
    Splits NSL-KDD dataset into features and labels. Returns a tuple with the two Pandas DataFrame objects.
    '''
    dataset = pd.read_csv(file_location, sep=",")
    dataset = dataset.drop(columns=['41_num_outbound_cmds'])
    return (dataset.iloc[:, 0:4], dataset.iloc[:, -2:-1]) if small_test else (dataset.iloc[:, 0:-2], dataset.iloc[:, -2:-1])


def build_feature_columns(dataset):
    ''' Generates numeric feature columns for continious features and categorical columns for features represented by a string.     

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

            # Sorting is important since sets have no special ordering
            # which would result in different one-hot encodings on every run
            categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key=header,
                vocabulary_list=sorted(set(dataset[header]))
            )
            feature_columns.append(tf.feature_column.indicator_column(categorical_column))
        else:
            feature_columns.append(tf.feature_column.numeric_column(key=header))

    return feature_columns

def get_full():
    return build_kdd_dataset(FULL_TRAIN_PATH, FULL_TEST_PATH)

def get_2c():
    return build_kdd_dataset(TWOC_TRAIN_PATH, TWOC_TEST_PATH)

def get_5c():
    return build_kdd_dataset(FIVEC_TRAIN_PATH, FIVEC_TEST_PATH)

def load_KDD(selector):
    if selector not in [1,2,3]: 
        raise ValueError('"selector" paramter must be 1, 2 or 3. You provided {}.'.format(selector))
    
    if selector == 1: return get_full()
    elif selector == 2: return get_2c()
    elif selector == 3: return get_5c()

