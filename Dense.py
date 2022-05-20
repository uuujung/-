from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import tensor_shape

class layer(Layer):
    
    def __init__(self, units, activation = None, initializer = 'glorot_uniform', trainable = True, dynamic = True):

        super(layer, self).__init__()

        self.units = units
        self.activation = activations.get(activation)
        self.initializer = initializers.get(initializer)


    def build(self, input_shape):
        
        last_dim = tensor_shape.dimension_value(input_shape[-1])

        self.W = self.add_weight('W',
                                      shape = (last_dim, self.units),
                                      initializer = self.initializer,
                                      dtype = 'float32',
                                      trainable = True)
        
        self.b = self.add_weight('b',
                                    shape = (self.units),
                                    initializer = initializers.get('zeros'),
                                    dtype = 'float32',
                                    trainable = True)        
        

    def call(self, inputs):
        
        outputs = math_ops.matmul(inputs, self.W) + self.b

        if self.activation is not None:
              return self.activation(outputs) 

        return outputs

