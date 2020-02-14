from Generator import Generator
from DiscriminatorPatch import Discriminator
from keras import layers
from keras import Model
from keras import optimizers
from keras import losses
from keras import backend as K

class CombinedModel():
    def __init__(self, learning_rate = 0.0004):
        # These variables are defined to make model changes much quicker
        self.number_of_filters = 2
        self.filter_size = 1024
        self.batch_size = 16
        self.sample_rate = 16384
        self.dilation_rate = 2
        self.input_shape = (16384, 1)

        self.initlr = learning_rate
        self.genOptimizer = optimizers.Adam(0.0004, 0.5)
        self.discOptimizer = optimizers.Adam(0.0004, 0.5)

        self.disc_patch = (1, 1)
        self.downsampling_rate = 2

        self.genModel = Generator()
        self.generator = self.genModel.BuildGenerator(self.input_shape, self.number_of_filters, self.filter_size,
                                                      self.dilation_rate, self.downsampling_rate)
        self.generator.compile(optimizer=self.genOptimizer, loss='mse', metrics=['accuracy'])

        self.discModel = Discriminator()
        self.discriminator = self.discModel.BuildDiscriminator(self.input_shape, self.number_of_filters,
                                                               self.filter_size, self.dilation_rate,
                                                               self.downsampling_rate * 2)
        self.discriminator.compile(optimizer=self.discOptimizer, loss = 'mse', metrics=['accuracy'])

        # Define inputs of combined model
        dry_wave = layers.Input(shape=self.input_shape)
        wet_wave = layers.Input(shape=self.input_shape)

        # Generate a fake version of the wet waveforme
        fake_wave = self.generator(dry_wave)

        # Discriminator of combined model will not be trainable
        self.discriminator.trainable = True

        # Discriminator should make a prediction for the fake audio waveforme
        prediction = self.discriminator([fake_wave, dry_wave])
        self.combined = Model(inputs=[wet_wave, dry_wave], outputs=[prediction, fake_wave])

        #train generator with L1Loss and Discriminator with
        self.combined.compile(loss=['mse', 'mae'], loss_weights=[1, 100], optimizer=self.genOptimizer)

    def get_model(self):
        return self.combined

    def get_input_shape(self):
        return self.input_shape

    def save_model_weights(self, path):
        self.combined.save_weights(path)

    def load_model_weights(self, path):
        self.combined.load_weights(path)

    def change_generator_learning_rate(self, learning_rate_value):
        K.set_value(self.generator.optimizer.lr, learning_rate_value)

    def change_discriminator_learning_rate(self, learning_rate_value):
        K.set_value(self.discriminator.optimizer.lr, learning_rate_value)