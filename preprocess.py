import aifc
import numpy as np
import os
from tqdm import tqdm
import torch
import torchaudio
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import math

"""
The following script contains the main functions for data preprocessing
We can choose to realize the preprocessing with aifc or torchaudio libraries.
The script also provides a function for generating the audio signals' melspectrograms
as images.

"""

# reads an audio signal from a .aiff file
# returns a numpy array
def load_aiff(path):
    
    sample = aifc.open(path)
    n_frames = sample.getnframes()
    byte_buffer =  sample.readframes(n_frames)
    # need to do .byteswap as aifc loads / converts to a bytestring in
    # MSB ordering, but numpy assumes LSB ordering.
    array =np.frombuffer(byte_buffer, dtype = np.int16).byteswap()
    return array

    
# reads audio signals from a folder, annotations dataframe
# returns dataset and labels as numpy arrays
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

# reads audio signals from a directory, annotations dataframe
# returns dataset and labels as torch tensors
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


# scales an array over a min max window
def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

# computes melspectrogram of a signal converted to log scale (decibel), and saves it as an image
def spectrogram_image(audio_file, out, hop_length, n_mels):
    
    y, sr = librosa.load(audio_file, sr = None)
    # use log-melspectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                            n_fft=hop_length*2, hop_length=hop_length)
    log_S = librosa.power_to_db(S, ref=np.max)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(log_S, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255-img # invert : make black=more energy
    img = Image.fromarray(img)
    # save as PNG
    img.save(out)


# computes all melspectrograms from the main directory ,and
# saves them in an image directory,  given a desired image shape

def get_images(data_path, img_dir, input_shape, wave_length ):
    
    # settings
    n_mels = input_shape[1] # number of bins in spectrogram. height of image
    time_steps = input_shape[0] # number of time-steps : width of image
    
    # number of samples per time-step in spectrogram
    # including all the wave implies hop_lenth = wave_length/ time_steps
    
    hop_length = math.ceil(wave_length/time_steps) 
    
    
    train_dir = os.listdir(data_path)
    pbar = tqdm(len(train_dir))
    for audio_filename in train_dir:
            audio_path = os.path.join(data_path, audio_filename)
            image_fname = audio_filename.split('.')[0] + '.png'
            image_fname = img_dir + '/' + image_fname
            spectrogram_image(audio_path, image_fname, hop_length, n_mels)
            pbar.update(1)



if __name__ == "__main__" :
    
        
    data_raw_path = './data/train/'
    sample = data_raw_path + "train10.aiff"
    
    # Get wave and sample rate
    y, sr = librosa.load(sample, sr = None) 
    
    # Make and display a mel-scaled power (energy-squared) spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    
    # Convert to log scale (dB). We'll use the peak power as reference.
    log_S = librosa.power_to_db(S, ref=np.max)
    
    # Display the spectrogram on a mel scale
    # sample rate is used to render the time axis
    librosa.display.specshow(log_S, sr=sr,  x_axis='time', y_axis='mel')

    fig = plt.gcf()
    fig.savefig('./results/spectrogram/mel.png')