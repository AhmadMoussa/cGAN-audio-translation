from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import add

class WeightedResidual(Layer):
    def __init__(self, **kwargs):
        super(WeightedResidual, self).__init__(**kwargs)

    def build(self, input_shape):

        self._x = K.variable(0.5)
        self.trainable_weights = [self._x]

        super(WeightedResidual, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        A, B = x
        result = add([self._x*A ,(1-self._x)*B])
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0]