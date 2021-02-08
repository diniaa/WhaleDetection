import aifc
import numpy as np
import os
from tqdm import tqdm
import torch
import torchaudio


def load_aiff(path):
    
    sample = aifc.open(path)
    n_frames = sample.getnframes()
    byte_buffer =  sample.readframes(n_frames)
    array =np.frombuffer(byte_buffer, dtype = np.int16).byteswap()
    # need to do .byteswap as aifc loads / converts to a bytestring in
    # MSB ordering, but numpy assumes LSB ordering.
    return array

    

def load_dataset(data_path, annotations):
    X, Y = [],[]
    train_dir = os.listdir(data_path)
    pbar = tqdm(len(train_dir))
    for aiff_file in train_dir:
        full_path = os.path.join(data_path, aiff_file)
        x = load_aiff(full_path)
        y = annotations.query("clip_name == {}".format("".join(["'", aiff_file, "'"]))).label.item()
        X.append(x)
        Y.append(y)
        pbar.update(1)
        
        
    X = np.stack(X)
    Y = np.array(Y)
    return X,Y

def load_with_torchaudio(data_path, annotations):
    
    X, Y = [],[]
    train_dir = os.listdir(data_path)
    pbar = tqdm(len(train_dir))
    for aiff_file in train_dir:
        full_path = os.path.join(data_path, aiff_file)
        x, sample_rate = torchaudio.load(full_path)
        assert sample_rate == 2000
        y = annotations.query("clip_name == {}".format("".join(["'", aiff_file, "'"]))).label.item()
        X.append(x)
        Y.append(y)
        pbar.update(1)
    
    return torch.stack(X), torch.FloatTensor(Y)