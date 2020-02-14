import matplotlib.pyplot as plt
import numpy as np

audio = np.load("high_pass_dataset/train/bass_acoustic_000-024-025_chunk_3.npy")
print(audio[16384:].shape)