import json
from pathlib import Path
from collections import OrderedDict
from itertools import repeat
import pandas as pd
import os
import numpy as np
from glob import glob
import math
import sys

def load_folds_data_shhs(np_data_path, n_folds):
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    r_p_path = r"utils/r_permute_shhs.npy"
    r_permute = np.load(r_p_path)
    npzfiles = np.asarray(files , dtype='<U200')[r_permute]
    train_files = np.array_split(npzfiles, n_folds)
    folds_data = {}
    for fold_id in range(n_folds):
        subject_files = train_files[fold_id]
        training_files = list(set(npzfiles) - set(subject_files))
        folds_data[fold_id] = [training_files, subject_files]
    return folds_data

def load_folds_data(np_data_path, n_folds):
    print(np_data_path) #prints /dataverse_files
    
    # Get the current directory of the script (utils directory)
    #script_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up one level to reach the larger directory
    #larger_directory = os.path.dirname(script_dir)

    # Construct the full path to the dataverse_files directory
    #dataverse_files_path = os.path.join(larger_directory, "dataverse_files")

    # Now you can use dataverse_files_path to access the folder
    #print("Data directory:", dataverse_files_path)
    
    if not os.path.exists(np_data_path):
        print(f"Data path '{np_data_path}' does not exist.")
        sys.exit()
    else:
        files = sorted(glob(os.path.join(np_data_path, "*.npz")))

    '''
    search_pattern = os.path.join(np_data_path, "*.npz")
    matched_files = glob(search_pattern)
    sorted_files = sorted(matched_files)
    files = sorted_files
    '''
    print("Number of files:", len(files))
    
    ''' 
    if "78" in np_data_path:
    r_p_path = r"utils/r_permute_78.npy"
    else:
    r_p_path = r"utils/r_permute_20.npy"
    '''
    r_p_path = r"utils/r_permute_20.npy"

    if os.path.exists(r_p_path):
        r_permute = np.load(r_p_path)
    else:
        print ("============== ERROR =================")

    print(type(r_permute))
    array_length = len(r_permute)
    print("Length of r_permute:", array_length)
    print("rperm shape: ", r_permute.shape)

    files_dict = dict()
    for i in files:
        file_name = os.path.split(i)[-1] 
        file_num = file_name[3:5]
        if file_num not in files_dict:
            files_dict[file_num] = [i]
        else:
            files_dict[file_num].append(i)

    files_pairs = list(files_dict.values())

    # Shuffle the order of the groups based on r_permute
    shuffled_indices = np.arange(len(files_pairs))
    shuffled_indices = shuffled_indices[r_permute]
    files_pairs = [files_pairs[i] for i in shuffled_indices]

    # Continue with the rest of your code
    train_files = [files_pairs[i] for i in range(n_folds)]  # Split the shuffled list into folds

    folds_data = {}
    for fold_id in range(n_folds):
        subject_files = [item for sublist in train_files[fold_id] for item in sublist]
        files_pairs2 = [item for sublist in files_pairs for item in sublist]
        training_files = list(set(files_pairs2) - set(subject_files))
        folds_data[fold_id] = [training_files, subject_files]

    # Return the folds_data dictionary
    return folds_data




    '''
    files_dict = dict()
    for i in files:
        file_name = os.path.split(i)[-1] 
        file_num = file_name[3:5]
        if file_num not in files_dict:
            files_dict[file_num] = [i]
        else:
            files_dict[file_num].append(i)
    files_pairs = []
    for key in files_dict:
        files_pairs.append(files_dict[key])
    files_pairs = np.array(files_pairs)
    print("fp shape: ", files_pairs.shape)
    files_pairs = files_pairs[r_permute]

    train_files = np.array_split(files_pairs, n_folds)
    folds_data = {}
    for fold_id in range(n_folds):
        subject_files = train_files[fold_id]
        subject_files = [item for sublist in subject_files for item in sublist]
        files_pairs2 = [item for sublist in files_pairs for item in sublist]
        training_files = list(set(files_pairs2) - set(subject_files))
        folds_data[fold_id] = [training_files, subject_files]
    return folds_data
    '''


def calc_class_weight(labels_count):
    total = np.sum(labels_count)
    class_weight = dict()
    num_classes = len(labels_count)

    factor = 1 / num_classes
    mu = [factor * 1.5, factor * 2, factor * 1.5, factor, factor * 1.5] # THESE CONFIGS ARE FOR SLEEP-EDF-20 ONLY

    for key in range(num_classes):
        score = math.log(mu[key] * total / float(labels_count[key]))
        class_weight[key] = score if score > 1.0 else 1.0
        class_weight[key] = round(class_weight[key] * mu[key], 2)

    class_weight = [class_weight[i] for i in range(num_classes)]

    return class_weight


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
