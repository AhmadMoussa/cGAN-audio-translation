from FastYielder import Yielder
from Visualizer import visualize
import numpy as np
import glob
import os
import gc
import pickle
import pprint

import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

'''
    Takes as input a model and trains it.
    Allows different learning rate techniques
'''

class Trainer():
    def __init__(self, model, epochs, checkpoints_path = 'ModelCheckPoints/', load_model_bool = True, learning_rate = 0.0004):
        '''
            Model parameters:

            :param model:               the model to train
            :param epochs:              number of epochs to train for
            :param checkpoints_path:    location of model checkpoints
            :param load_model_bool:     boolean that controls loading a model or not
            :param learning_rate:       sets the learning rate
            :      start_epochs         begin from epoch 0 but if model is loaded begin from later epoch
        '''
        self.model = model
        self.load_model_bool = load_model_bool
        self.checkpoints_path = checkpoints_path
        self.start_epoch = 0
        self.epochs = epochs

        self.learning_rate = learning_rate

        # this makes print() print arrays on one line
        np.set_printoptions(linewidth=np.inf)

        '''
            training_data_loader:       dataloader that will yield batchers for training the model
            evaluation_data_loader:     dataloader that will yield batches for evaluating the model
        '''

        # Set up data loaders for training and evaluation data
        trainPath = "low_pass_dataset/train/"
        evalPath = "low_pass_dataset/tiny_eval/"
        self.trainLoader = Yielder(self.model.get_input_shape(), "train_index.npy", self.model.batch_size, self.model.sample_rate, trainPath,
                              shuffle=True)
        self.evalLoader = Yielder(self.model.get_input_shape(), "low_pass_tiny_eval_index.npy", 1, self.model.sample_rate, evalPath,
                             shuffle=False)

        # Discriminator targets
        self.valid = np.ones((self.model.batch_size, 1)) - 0.08
        self.invalid = np.zeros((self.model.batch_size, 1)) + 0.08

        if load_model_bool == True:
            self.load_model(checkpoints_path)

    def load_model(self, checkpoints_path):
        try:
            list_of_files = glob.glob(
                checkpoints_path + '*')  # * means all if need specific format then *.csv
            latest_file = max(list_of_files, key=os.path.getctime)
            self.start_epoch = int(latest_file[-7:-3]) + 1
            print(latest_file)
            self.model.combined.load_weights(latest_file)
            print("Loaded model")
        except:
            print("Something went wrong loading weights")



    def cyclical_learning_rate(self, upper_limit, lower_limit, increase):
        '''
            Function that controls the cyclical learning rate

            :param upper_limit:             highest learning rate
            :param lower_limit:             lowest learning rate
            :param increase:                step size of learning rate between upper and lower limit
            :return:                        None
        '''
        if self.initlr > self.upperlimit:
            self.increase = -0.00004
        elif self.initlr < self.lowerlimit:
            self.increase = 0.00004
        self.learning_rate += self.increase
        self.model.change_generator_learning_rate(self.learning_rate)
        self.model.change_discriminator_learning_rate(self.learning_rate)

        print("The learning rate is: ", self.initlr)


    def train_model(self):
        # Mini batch sizes
        number_of_training_batches = 100
        number_of_evaluation_batches = 10
        count = 0

        # Validation frequency
        eval_freq = 1

        d_running_loss = 0
        g_running_loss = 0
        for epoch in range(self.start_epoch, self.epochs):
            # Shuffle data on epoch end
            self.trainLoader.data_shuffler()
            #self.evalLoader.data_shuffler()

            for batch_i, (dry_batch, wet_batch) in enumerate(self.trainLoader.batch_yielder()):
                fake_wave = self.model.generator.predict(dry_batch)
                '''
                print("Discriminator prediction on real: ",
                      " ".join(map(str, self.model.discriminator.predict([wet_batch, dry_batch]))))
                print("Discriminator prediction on fake: ",
                      " ".join(map(str, self.model.discriminator.predict([fake_wave, dry_batch]))))
                '''
                d_loss_real = self.model.discriminator.train_on_batch([wet_batch, dry_batch], self.valid)
                d_loss_fake = self.model.discriminator.train_on_batch([fake_wave, dry_batch], self.invalid)
                d_loss = np.add(d_loss_fake, d_loss_real) * 0.5

                g_loss = self.model.combined.train_on_batch([wet_batch, dry_batch], [self.valid, wet_batch])
                print(
                    "(Epoch: {})(Training Batch: {})[Disc train loss: {}]   [Gen train loss: {}]".format(epoch, batch_i,
                                                                                                         d_loss,
                                                                                                         g_loss))

                count += 1
                writer.add_scalar('Loss/discriminator', torch.tensor(d_loss[0]), count)
                writer.add_scalar('Loss/generator', torch.tensor(g_loss[0]), count)
                d_running_loss += d_loss[0]
                g_running_loss += g_loss[0]
                count += 1
                if batch_i == (number_of_training_batches - 1):
                    break

            print("\n(Training Epoch: {})[Disc overall train loss: {}]   [Gen overall train loss: {}]\n".format(epoch,
                                                                                                 d_running_loss / (number_of_training_batches-1),
                                                                                                                g_running_loss / (number_of_training_batches-1)))
            writer.add_scalar('Loss/discriminator_running', torch.tensor(d_running_loss)/100, epoch)
            writer.add_scalar('Loss/generator_running', torch.tensor(g_running_loss)/100, epoch)
            d_running_loss = 0
            g_running_loss = 0
            # Save model
            self.model.save_model_weights(
                self.checkpoints_path + "/Epoch_{:04d}.h5".format(epoch))

            # Batch size larger than 1 doesn't make sense here
            for batch_i, (dry_batch, wet_batch) in enumerate(self.evalLoader.batch_yielder()):
                fake = self.model.generator.predict(dry_batch)
                visualize(epoch, batch_i, dry_batch[0], wet_batch[0], fake[0])

                if batch_i == (number_of_evaluation_batches - 1):
                    break

            # Finally collect garbage and clean
            n = gc.collect()
            print("Unreachable objects: ", n)
            print("Remaining Garbage: ")
            pprint.pprint(gc.garbage)

print("Please run from main script")
#This is the end of the file