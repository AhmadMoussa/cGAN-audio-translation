import numpy as np
import os

class Yielder():
    # Constructor
    def __init__(self, input_shape, filenames_path, yielder_batch_size, sample_rate, data_path, shuffle=True):
        # the shape of data will be (16384)
        self.input_shape = input_shape

        # a list of the training_data file names
        try:
            self.audio_filenames = np.load(filenames_path)
        except:
            print("Couldn't load filenames")

        self.audio_filenames_shuffle = self.audio_filenames

        # for our model this will be a length of 16384
        self.sample_rate = sample_rate

        # should maximally be 32
        self.yielder_batch_size = yielder_batch_size

        # location of training/evaluation samples
        self.data_path = data_path

        # Boolean that specifies if we should shuffle data after every epoch or not
        self.shuffle = shuffle

        # This function determines the number of batches

    def lenth(self):
        return int(np.floor(len(self.audio_filenames) / float(self.yielder_batch_size)))

    # yields a batch of data
    def batch_yielder(self):
        for i in range(self.lenth()):
            batch = self.audio_filenames_shuffle[i * self.yielder_batch_size:(i + 1) * self.yielder_batch_size]
            dry_batch = np.empty((self.yielder_batch_size, self.input_shape[0], self.input_shape[1]))
            wet_batch = np.empty((self.yielder_batch_size, self.input_shape[0], self.input_shape[1]))

            for j, file in enumerate(batch):
                audio = np.load(os.path.join(self.data_path, file))
                dry_batch[j,], wet_batch[j,] = audio[:16384].reshape((16384, 1)), audio[16384:32768].reshape((16384, 1))

            yield dry_batch, wet_batch

    # yields a batch of data but dry and wet are flipped
    def batch_yielder_inverse(self):
        lenth = self.lenth()
        for i in range(lenth):
            batch = self.audio_filenames_shuffle[i * self.yielder_batch_size:(i + 1) * self.yielder_batch_size]
            dry_batch, wet_batch = np.empty(
                (self.yielder_batch_size, self.input_shape[0], self.input_shape[1])), np.empty(
                (self.yielder_batch_size, self.input_shape[0], self.input_shape[1]))

            for j, file in enumerate(batch):
                audio = np.load(os.path.join(self.data_path, file))
                dry_batch[j,], wet_batch[j,] = audio[:16384].reshape(16384, 1), audio[16384:].reshape(16384, 1)

            yield wet_batch, dry_batch

    # this should be called at the end of the epoch to shuffle data
    def data_shuffler(self):
        np.random.shuffle(self.audio_filenames_shuffle)
