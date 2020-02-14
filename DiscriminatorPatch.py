from keras import layers
from keras import Model

class Discriminator():
    def __init__(self, batch_norm_momentum = 0.1, leaky_relu_alpha = 0.3):
        # These variables are defined to make model changes much quicker
        self.number_of_filters = 2
        self.filter_size = 1024
        self.dilation_rate = 4
        self.input_shape = (16384, 1)
        self.downsampling_rate = 4

        self.batch_norm_momentum = batch_norm_momentum
        self.leaky_relu_alpha = leaky_relu_alpha

    def getDiscriminator(self):
        return self.discriminator

    def compileDiscriminator(self, optimizer):
        self.discriminator.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

    def pool_conv_1d_first(self, in_put, filters, f_size, dr, ps):
        pc1df = layers.Conv1D(filters, f_size, padding='causal', dilation_rate=dr, use_bias = False)(in_put)
        pc1df = layers.BatchNormalization(momentum=self.batch_norm_momentum)(pc1df)
        pc1df = layers.LeakyReLU(alpha=self.leaky_relu_alpha)(pc1df)
        pc1df = layers.MaxPooling1D(pool_size=ps, strides=None, padding='valid', data_format='channels_last')(pc1df)
        return pc1df

    def pool_conv_1d(self, in_put, filters, f_size, dr, ps):
        pc1d = layers.Conv1D(filters, f_size, padding='causal', dilation_rate=dr)(in_put)
        pc1d = layers.BatchNormalization(momentum=self.batch_norm_momentum)(pc1d)
        pc1d = layers.LeakyReLU(alpha=self.leaky_relu_alpha)(pc1d)
        pc1d = layers.MaxPooling1D(pool_size=ps, strides=None, padding='valid', data_format='channels_last')(pc1d)
        return pc1d

    def last_out(self, in_put, filters, f_size, dr, ps):
        lo = layers.Conv1D(filters, f_size, padding='causal', dilation_rate=dr)(in_put)
        lo = layers.LeakyReLU(alpha=self.leaky_relu_alpha)(lo)
        lo = layers.MaxPooling1D(pool_size=ps, strides=None, padding='valid', data_format='channels_last')(lo)
        return lo

    def BuildDiscriminator(self, input_shape, number_of_filters, filter_size, dilation_rate, downsampling_rate):
        # Generator Layers
        d00 = layers.Input(input_shape)
        d01 = layers.Input(input_shape)
        dc = layers.Concatenate()([d00,d01])
        print("Input shape: ", dc.shape)

        first = self.pool_conv_1d_first(dc, number_of_filters, filter_size, dilation_rate * 16, int(downsampling_rate))
        print(first.shape)

        d1 = self.pool_conv_1d(first, number_of_filters * 2, filter_size, dilation_rate * 16,
                             int(downsampling_rate))
        print(d1.shape)

        d2 = self.pool_conv_1d(d1, number_of_filters * 2, filter_size, dilation_rate * 12, int(downsampling_rate))
        print(d2.shape)

        d3 = self.pool_conv_1d(d2, number_of_filters * 4, filter_size, dilation_rate * 8, int(downsampling_rate))
        print(d3.shape)

        d4 = self.pool_conv_1d(d3, number_of_filters * 4, filter_size, dilation_rate * 6, int(downsampling_rate))
        print(d4.shape)

        d5 = self.pool_conv_1d(d4, number_of_filters * 8, filter_size, dilation_rate * 4, int(downsampling_rate))
        print(d5.shape)

        d6 = self.pool_conv_1d(d5, number_of_filters * 8, filter_size, dilation_rate * 4, int(downsampling_rate))
        print(d6.shape)

        down_out = self.last_out(d6, 1, 2048, 1, 1)
        print(down_out.shape)

        out = layers.Flatten()(down_out)
        out = layers.Dense(2048, activation = 'relu')(out)
        out = layers.Dense(1, activation = 'relu')(out)
        return Model(inputs = [d00, d01], outputs = out)

from keras import optimizers
discOptimizer = optimizers.Adam(0.00001, 0.5)

disc = Discriminator()
disc.BuildDiscriminator((16384, 1), 2, 1024, 2, 4)
