from torch.multiprocessing import Process, Queue, Value, Pipe
import multiprocessing
import queue
import os

import pickle
import sys
from threading import Thread, Lock
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.decomposition import PCA

sys.path.append("./..")
import util

lock = Lock()

class CustomDataset(Dataset):

    def __init__(self, dataset, debug_mode=False, build_fn=None, num_workers=3, pca=None, params=None, cache_path=None, logger=None, debug=False):
        """
        Converts the KDD Dataset files into a PyTorch-Dataset object. The Dataset object is to be
        passeed to a PyTorch-DataLoader. The DataLoader then uses __getitem__ to retrieve batches
        in the form {'data': input_vector, 'label': attack_type}. Textual fields are one hot encoded.

        Args:
            dataset (DataFrame): A Pandas dataset that will be used
            debug_mode: If True only a small sample for debugging will be returned

            build_fn: A function that takes a single row as input and returns a dict with keys 'data' and 'label', both of which hold a list of numpy objects
            
        """
        self.dataset = dataset 

        self.params = params
        self.preprocessed = []

        self.build_fn = build_fn

        enter_build_fn_block = True
        if cache_path is not None:
            try: pickle_in = open(cache_path,"rb")
            except FileNotFoundError: pass
            else: 
                enter_build_fn_block = False
                self.preprocessed = pickle.load(pickle_in)
                pickle_in.close()

        if self.build_fn and enter_build_fn_block: 
            self.num_workers = num_workers
            self.messagePipes = {
                'proc_' + str(id): Pipe() for id in range(self.num_workers)
            }
            self.finished = Value('i',0)
            self._build(dataset)

            if cache_path is not None:
                pickle_out = open(cache_path,"wb")
                pickle.dump(self.preprocessed, pickle_out)
                pickle_out.close()
    
        # Creating the logger before the multiprocessing step
        # causes errors as spawned processes try to pickle it    
        self.logger = logger
        if not self.build_fn:
            if self.logger:
                self.logger.warning('No build function specified. __getitem__() will return the raw dataset.')
        else: 
            if self.logger:
                self.logger.debug('Build complete with {} failure(s)'.format(abs(len(self.preprocessed) - len(dataset))))
        
        if pca is not None: self._do_pca(pca)

    def _do_pca(self, n_components):
        msg = 'Performing PCA with {} components.'.format(n_components)
        if self.logger: self.logger.info(msg)
        else: print(msg)

        data_only = [entry['data'].tolist() for entry in self.preprocessed]
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(data_only)
        for i, elem in enumerate(transformed): 
            self.preprocessed[i]['data'] = torch.tensor(elem)

        msg = 'Retained variance: {:.4f}'.format(pca.explained_variance_ratio_.cumsum()[-1])
        if self.logger: self.logger.info(msg)
        else: print(msg)

    def __call__(self, batch_size, shuffle=False, num_workers=0, pin_memory=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    def __len__(self):
        if self.build_fn: return len(self.preprocessed)
        else: return len(self.dataset)

    def _helper(self,dataset, sender_end, position):
        '''
          Args: 
            dataset (DataFrame) - part of the whole dataset to be preprocessed by a worker process
        '''
        every_n = 10000
        batch = []
        for count, (_, row) in enumerate(dataset.iterrows()):
            
            batch.append(self.build_fn(row, self.params))

            if count % every_n == 0:
                sender_end.send(batch)
                batch = []
                
            count += 1

        sender_end.send(batch)
        sender_end.send(-1)

        with self.finished.get_lock():
            self.finished.value += 1

    def _receiver(self, receiver_end, id):
        finished = 0
        batch_2 = []
        while True:
            batch = receiver_end.recv()
            if batch == -1:
                break

            for item in batch:

                label_tensor = torch.tensor(item['label'])
                data_tensor = torch.tensor(item['data'])

                batch_2.append({
                    'data':data_tensor,
                    'label':label_tensor
                })

                if len(batch_2) >= 200:
                    with lock:
                        self.preprocessed.extend(batch_2)
                    batch_2 = []

        with lock:
            self.preprocessed.extend(batch_2)

    def _single_thread_build(self, dataset):
        for _, row in dataset.iterrows():
            item = self.build_fn(row, self.params)

            label_tensor = torch.tensor(item['label'])
            data_tensor = torch.tensor(item['data'])

            self.preprocessed.append({
                'data':data_tensor,
                'label':label_tensor
            })


    def _build(self,dataset):
        if self.num_workers == 1: 
            self._single_thread_build(dataset)
        else:
            # Divide the dataset into equal parts and send to the worker processes
            processes = []
            threads = []
            ds_size = len(dataset)
            chunk_size = ds_size // self.num_workers

            for i in range(self.num_workers): 
                if i == self.num_workers - 1:
                    ds_chunk = dataset.iloc[i*chunk_size:]
                else:
                    ds_chunk = dataset.iloc[i*chunk_size:i*chunk_size + chunk_size]
                # thread = Thread(target=self._receiver, args=(self.messagePipes['proc_' + str(i)][0],i))
                process = Process(target=self._helper, name='helper_' + str(i), args=(ds_chunk, self.messagePipes['proc_' + str(i)][1], i))
                processes.append(process)
                # thread.start()
                process.start()
                

            for i in range(self.num_workers):
                thread = Thread(target=self._receiver, args=(self.messagePipes['proc_' + str(i)][0],i))
                threads.append(thread)
                thread.start()

            for process in processes: 
                process.join()
            
            for thread in threads:
                thread.join()

    def output_size(self):
        return len(set([elem['label'][0].item() for elem in self.preprocessed]))

    def input_size(self):
        return len(self.preprocessed[0]['data'])

    def __getitem__(self, idx):
        if self.build_fn: return self.preprocessed[idx]
        elif isinstance(self.dataset,pd.DataFrame):
            return self.dataset.iloc[idx,:]
        else: return self.dataset[idx]

def load(dataset, batch_size, shuffle=True):
    return dataset(batch_size, shuffle=shuffle)   

def transform_fn(row, params):
        attack_header = params['attack_header']
        protocol_header = params['protocol_header']
        service_header = params['service_header']
        flag_header = params['flag_header']

        attack_dict = params['attack_dict']
        protocol_dict = params['protocol_dict']
        service_dict = params['service_dict']
        flag_dict = params['flag_dict']

        protocol = util.one_hot_vector(protocol_dict[row[protocol_header]], len(protocol_dict), dtype=np.float64)
        service = util.one_hot_vector(service_dict[row[service_header]], len(service_dict), dtype=np.float64)
        flag = util.one_hot_vector(flag_dict[row[flag_header]], len(flag_dict), dtype=np.float64)

        all_else = row[3:-2].values.astype(np.float64)

        data = np.concatenate([protocol,service,flag,all_else])

        attack = [attack_dict[row[attack_header]]]
        
        return {'data':data, 'label':attack}

def load_KDD(selector, pca=None, debug_mode=False, logger=None):
    if selector not in [1,2,3]: 
        raise ValueError('"selector" paramter must be 1, 2 or 3. You provided {}.'.format(selector))
        
    TRAIN_DIR_FULL = 'Dataset/Train/KDDTrain.csv'
    TEST_DIR_FULL = 'Dataset/Test/KDDTest.csv' 

    TRAIN_DIR_TWOC = 'Dataset/Train/KDDTrain_2c.csv'
    TEST_DIR_TWOC = 'Dataset/Test/KDDTest_2c.csv'

    TRAIN_DIR_FIVEC = 'Dataset/Train/KDDTrain_5c.csv'
    TEST_DIR_FIVEC = 'Dataset/Test/KDDTest_5c.csv'

    ##########

    TRAIN_DIR_FULL_CACHE = 'Cache/PyTorch/KDDTrain.pickle'
    TEST_DIR_FULL_CACHE = 'Cache/PyTorch/KDDTest.pickle' 

    TRAIN_DIR_TWOC_CACHE = 'Cache/PyTorch/KDDTrain_2c.pickle'
    TEST_DIR_TWOC_CACHE = 'Cache/PyTorch/KDDTest_2c.pickle'

    TRAIN_DIR_FIVEC_CACHE = 'Cache/PyTorch/KDDTrain_5c.pickle'
    TEST_DIR_FIVEC_CACHE = 'Cache/PyTorch/KDDTest_5c.pickle'

    selector -= 1

    datasets = [
        (TRAIN_DIR_FULL,TEST_DIR_FULL),
        (TRAIN_DIR_TWOC,TEST_DIR_TWOC),
        (TRAIN_DIR_FIVEC,TEST_DIR_FIVEC)
    ]

    cache = [
        (TRAIN_DIR_FULL_CACHE,TEST_DIR_FULL_CACHE),
        (TRAIN_DIR_TWOC_CACHE,TEST_DIR_TWOC_CACHE),
        (TRAIN_DIR_FIVEC_CACHE,TEST_DIR_FIVEC_CACHE)
    ]

    training_path, testing_path = datasets[selector]
    training_cache, testing_cache = cache[selector]

    attack_header = 'attack'
    protocol_header = '01_protocol'
    service_header = '02_service'
    flag_header = '03_flag'

    train = pd.read_csv(training_path).drop(columns=['41_num_outbound_cmds'])
    test = pd.read_csv(testing_path).drop(columns=['41_num_outbound_cmds'])

    train = train.reindex(sorted(train.columns), axis=1)
    test = test.reindex(sorted(test.columns), axis=1)

    unique_labels = util.get_uniques(train[attack_header],test[attack_header])

    unique_protocol = util.get_uniques(train[protocol_header],test[protocol_header])
    unique_service = util.get_uniques(train[service_header],test[service_header])
    unique_flg = util.get_uniques(train[flag_header],test[flag_header])

    attack_dict = util.get_value_to_num_dict(unique_labels, sort=True)

    # Set baseline class to be 0
    for key in attack_dict: 
        value = attack_dict[key]
        if value == 0:
            attack_dict[key] = attack_dict['normal']
            attack_dict['normal'] = 0
            break 

    protocol_dict = util.get_value_to_num_dict(unique_protocol, sort=True)
    service_dict = util.get_value_to_num_dict(unique_service, sort=True)
    flag_dict = util.get_value_to_num_dict(unique_flg, sort=True)

    params = {
        'attack_dict':attack_dict,
        'attack_header':attack_header,
        'protocol_dict': protocol_dict,
        'protocol_header': protocol_header,
        'service_dict': service_dict,
        'service_header': service_header,
        'flag_dict': flag_dict,
        'flag_header': flag_header,
    }

    if logger: logger.info('Loading training data...')
    ds_train = CustomDataset(train, build_fn=transform_fn, pca=pca, params=params, num_workers=3, cache_path=training_cache, logger=logger)

    if logger: logger.info('Loading test data...')
    ds_test = CustomDataset(test, build_fn=transform_fn, pca=pca, params=params, num_workers=1, cache_path=testing_cache, logger=logger)

    return ds_train, ds_test, len(unique_labels)
