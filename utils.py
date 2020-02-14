import time
import librosa
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

def print_index(index_path):
    print(len(np.load(index_path)))

'''
    Creates a numpy array that contains the indexes of the dataset which allows for faster loading
'''
def generate_index(data_path, index_path):
    file_names = os.listdir(data_path)
    np.save(index_path, file_names)

'''
    Applies a low pass filter to a certain signal
    
    Tentative values for the filter:
        sr = 16384
        cutoff = 0.04
        b = 0.15
'''
def lowPassFilter(filter_cutoff, b, signal):
    N = int(np.ceil((4 / b)))
    if not N % 2:
        N += 1
    n = np.arange(N)

    sinc_func = np.sinc(2 * filter_cutoff * (n - (N - 1) / 2.))
    window = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
    sinc_func = sinc_func * window
    sinc_func = sinc_func / np.sum(sinc_func)
    return np.convolve(signal, sinc_func)

'''
    cuts up audio files to 1 second chunks depending on the samplerate and saves them as .npy arrays
    if audio can't be perfectly split up into 1 second it will pad it as well
'''
def splitter(read_path, save_path, sample_rate, cutoff, b):
    '''

    :param read_path:       where to read files from
    :param save_path:       where to save output files to
    :param sample_rate:     samplerate to load files at
    :param cutoff:          cutoff frequency
    :param b:               filter resonance
    :return:                None
    '''
    for root, dirs, files in os.walk(read_path, topdown=True):
        for file_name in files:

                audio,  sr = librosa.core.load(file_name, sr = sample_rate)
                # find remaining length and pad with zeroes
                remainder = len(audio) % 16384
                pad = np.zeros(16384 - remainder)
                audio = np.concatenate((audio, pad))
                audio_length = len(audio) / 16384

                for i in range(0,int(audio_length)):
                    chunk = audio[i * sr: (i + 1) * sr]
                    np.save(save_path + file_name[:-3] + "_chunk_" + str(i), chunk)
                    #librosa.output.write_wav(save_path + file_name[:-3] + "_chunk_" + str(i) + ".wav",chunk, 16384)



