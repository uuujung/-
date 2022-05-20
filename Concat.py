from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras import initializers
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import tensor_shape


class layer(Layer):
    
    def __init__(self, output_dim, initializer = 'glorot_uniform', trainable = True, dynamic = True):

        super(layer, self).__init__()

        self.output_dim = output_dim
        self.initializer = initializers.get(initializer)

        
    def build(self, input_shape):

        self.input_dim = tensor_shape.dimension_value(input_shape[-1])

        self.forward_kernel = self.add_weight('forward_kernel',
                                      shape = (self.input_dim, self.output_dim),
                                      initializer = self.initializer,
                                      dtype = 'float32',
                                      trainable = True)
        
        self.backward_kernel = self.add_weight('backward_kernel',
                                      shape = (self.input_dim, self.output_dim),
                                      initializer = self.initializer,
                                      dtype = 'float32',
                                      trainable = True)


    def call(self, forward_inputs, backward_inputs):
        
        outputs = math_ops.matmul(forward_inputs, self.forward_kernel) + math_ops.matmul(backward_inputs, self.backward_kernel)
        
        return outputs

