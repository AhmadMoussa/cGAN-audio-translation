from keras import layers
from keras import Model

class Generator():
    def __init__(self, leaky_relu_alpha = 0.3, dropout_rate = 0.1, batch_norm_momentum = 0.1):
        # These variables are defined to make model changes much quicker
        self.number_of_filters = 2
        self.filter_size = 1024
        self.dilation_rate = 8
        self.input_shape = (16384, 1)
        self.downsampling_rate = 4
        self.leaky_relu_alpha = leaky_relu_alpha
        self.dropout_rate = dropout_rate
        self.batch_norm_momentum = batch_norm_momentum

    def get_generator(self):
        return self.generator

    def compile_generator(self, optimizer):
        self.generator.compile(optimizer = optimizer, loss='mse', metrics=['accuracy'])

    def pool_conv_1d_first(self, in_put, filters, f_size, dr, ps):
        pc1df = layers.Conv1D(filters, f_size, padding='causal', dilation_rate=dr, use_bias = False)(in_put)
        pc1df = layers.BatchNormalization(momentum = self.batch_norm_momentum)(pc1df)
        pc1df = layers.LeakyReLU(alpha=self.leaky_relu_alpha)(pc1df)
        pc1df = layers.MaxPooling1D(pool_size=ps, strides=None, padding='valid', data_format='channels_last')(pc1df)
        return pc1df

    def pool_conv_1d(self, in_put, filters, f_size, dr, ps):
        pc1d = layers.Conv1D(filters, f_size, padding='causal', dilation_rate=dr, use_bias = False)(in_put)
        pc1d = layers.BatchNormalization(momentum = self.batch_norm_momentum)(pc1d)
        pc1d = layers.LeakyReLU(alpha = self.leaky_relu_alpha)(pc1d)
        pc1d = layers.MaxPooling1D(pool_size=ps, strides=None, padding='valid', data_format='channels_last')(pc1d)
        pc1d = layers.Dropout(rate = self.dropout_rate)(pc1d)
        return pc1d

    def de_conv_1d(self, in_put, skip_in, filters, f_size, dr, upsampling_rate):
        dc1d = layers.UpSampling1D(size=upsampling_rate)(in_put)
        dc1d = layers.Conv1D(filters, f_size, padding='causal', dilation_rate=dr, use_bias = False)(dc1d)
        dc1d = layers.BatchNormalization(momentum = self.batch_norm_momentum)(dc1d)
        dc1d = layers.LeakyReLU(alpha=self.leaky_relu_alpha)(dc1d)

        dc1d = layers.Concatenate()([dc1d, skip_in])        #addupd = WeightedResidual()([prelu, skip_in])
        dc1d = layers.Dropout(rate = self.dropout_rate)(dc1d)
        return dc1d

    # end deconv is an upsampling layer that doesn't receive a skip connection
    def de_conv_1d_end(self, in_put, skip_in, filters, f_size, dr, upsampling_rate):
        us = layers.UpSampling1D(size=upsampling_rate)(in_put)
        convupd = layers.Conv1D(filters, f_size, padding='causal', dilation_rate=dr, use_bias = True, activation='tanh')(us)
        return convupd

    def BuildGenerator(self, input_shape, number_of_filters, filter_size, dilation_rate , downsampling_rate):
        # Generator Layers
        d0 = layers.Input(input_shape)
        print("Input shape: ", d0.shape)

        first = self.pool_conv_1d_first(d0, number_of_filters, filter_size, dilation_rate * 16, int(downsampling_rate ))
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

        d7 = self.pool_conv_1d(d6, number_of_filters * 16, filter_size, dilation_rate * 2, int(downsampling_rate))
        print(d7.shape)

        d8 = self.pool_conv_1d(d7, number_of_filters * 16, filter_size, dilation_rate * 2, int(downsampling_rate))
        print(d8.shape)

        d9 = self.pool_conv_1d(d8, number_of_filters * 32, filter_size, dilation_rate, int(downsampling_rate))
        print(d9.shape)

        d10 = self.pool_conv_1d(d9, number_of_filters * 64, filter_size, dilation_rate, int(downsampling_rate))
        print(d10.shape)

        u9 = self.de_conv_1d(d10, d9, number_of_filters * 32, filter_size, dilation_rate, downsampling_rate)
        print(u9.shape)

        u8 = self.de_conv_1d(u9, d8, number_of_filters * 16, filter_size, dilation_rate * 2, downsampling_rate)
        print(u8.shape)

        u7 = self.de_conv_1d(u8, d7, number_of_filters * 16, filter_size, dilation_rate * 2, downsampling_rate)
        print(u7.shape)

        u6 = self.de_conv_1d(u7, d6, number_of_filters * 8, filter_size, dilation_rate * 4, downsampling_rate)
        print(u6.shape)

        u5 = self.de_conv_1d(u6, d5, number_of_filters * 8, filter_size, dilation_rate * 4, downsampling_rate)
        print(u5.shape)

        u4 = self.de_conv_1d(u5, d4, number_of_filters * 4, filter_size, dilation_rate * 6, downsampling_rate)
        print(u4.shape)

        u3 = self.de_conv_1d(u4, d3, number_of_filters * 4, filter_size, dilation_rate * 8, downsampling_rate)
        print(u3.shape)

        u2 = self.de_conv_1d(u3, d2, number_of_filters * 2, filter_size, dilation_rate * 12, downsampling_rate)
        print(u2.shape)

        u1 = self.de_conv_1d(u2, d1, number_of_filters * 2, filter_size, dilation_rate * 16, downsampling_rate)
        print(u1.shape)

        last = self.de_conv_1d(u1, first, number_of_filters, filter_size, dilation_rate * 16, downsampling_rate)
        print(last.shape)

        final = self.de_conv_1d_end(last, d0, 1,int( filter_size /16 ), int(dilation_rate/2), downsampling_rate)
        print("final", final.shape)

        return Model(inputs=d0, outputs=final)