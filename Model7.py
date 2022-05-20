import tensorflow as tf
#from tensorflow.python.ops import math_ops

import Dense
import LSTM
import Concat
import AttentionLSTM

#from tensorflow.python.keras.engine.base_layer import Layer


class construct(tf.keras.Model):
    
    def __init__(self, in_dim, key_dim, out_dim, d_model, num_layers, backward = True):

        super(construct, self).__init__()

        self.num_layers = num_layers
        self.backward = backward
        
        self.Wi = self.add_weight(shape = (in_dim, d_model),
                                          initializer = 'glorot_uniform', dtype = 'float32', name = 'Wi')

        self.Wk = self.add_weight(shape = (key_dim, d_model),
                                          initializer = 'glorot_uniform', dtype = 'float32', name = 'Wk')
        
        self.Wt = self.add_weight(shape = (key_dim, d_model),
                                          initializer = 'glorot_uniform', dtype = 'float32', name = 'Wt')
        

        self.Wo = self.add_weight(shape = (out_dim, d_model),
                                          initializer = 'glorot_uniform', dtype = 'float32', name = 'Wo')

        # input encoder
        self.in_enrnn = LSTM.layer(d_model, False)

        # key encoder
        self.key_enrnn = LSTM.layer(d_model, False)
        
        
        self.tkey_enrnn = LSTM.layer(d_model, False)

        # key encoder
        self.out_enrnn = [LSTM.layer(d_model, False) for _ in range(self.num_layers)]


        # key - in docoder
        self.key_in_dernn = AttentionLSTM.layer(d_model, False)

        # key dense
        self.key_in_deodense = Dense.layer(key_dim)


        # output - in docoder
        self.out_in_dernn = AttentionLSTM.layer(d_model, False)
        
        # output - key docoder
        self.out_key_dernn = AttentionLSTM.layer(d_model, False)

        # decoder concat
        self.out_key_in_concat = Concat.layer(d_model)

        # output dense
        self.out_key_in_deodense = Dense.layer(out_dim)
        

    def call(self, inputs, tkeys, keys, outputs, init_states):
        
        # input encoder
        #print(inputs.shape, self.Wi.shape)
        in_inputs = tf.einsum('bsi, id -> bsd', inputs, self.Wi) #배치,윈도우,d_model
        #print('in_inpusts',in_inputs.shape)
    
        _, in_enoutputs = self.in_enrnn(in_inputs, init_states) # 배치, 윈도우, d_model
        #print('in_enoutputs',in_enoutputs.shape)
        
        # key encoder
        #print(tkeys.shape, self.Wk.shape)
        tkey_inputs = tf.einsum('bst, td -> bsd', tkeys, self.Wt) #배치,윈도우,d_model
        _, tkey_enoutputs = self.tkey_enrnn(tkey_inputs, init_states) #배치, 윈도우, d_model
        
        key_inputs = tf.einsum('bsk, kd -> bsd', keys, self.Wk)
        #print('key_inputs', key_inputs.shape)
        _, key_enoutputs = self.key_enrnn(key_inputs, init_states) #배치, 윈도우, d_model
        #print('key_enoutputs', key_enoutputs.shape)
        #print(last_state_.shape, key_enoutputs_.shape)
        

        # output encoder
        #print(tkeys.shape, self.Wk.shape)
        out_inputs = tf.einsum('bso, od -> bsd', outputs, self.Wo) #배치, 윈도우, d_model
        #print('out_inputs',out_inputs.shape)

        out_enoutputs = out_inputs
        for i in range(self.num_layers):
            _, out_enoutputs = self.out_enrnn[i](out_enoutputs, init_states) #배치 윈도우 d_model
        #print('out_enoutputs', out_enoutputs.shape)


        # key - in decoder
        key_in_deinputs = (key_enoutputs, in_enoutputs)
        key_in_deoutputs = self.key_in_dernn(key_in_deinputs, init_states) #enoutputs_[:, :, -1, :]) # 배치, 윈도우 d_model
        #print('key_in_deoutputs', key_in_deoutputs.shape)
        
        # key logits
        key_in_outputs_ = self.key_in_deodense(key_in_deoutputs) #배치, 윈도우, in_dim
        #print('key_in_outputs_', key_in_outputs_.shape)


        # out - in decoder
        out_in_deinputs = (out_enoutputs, in_enoutputs)
        out_in_deoutputs = self.out_in_dernn(out_in_deinputs, init_states) #enoutputs_[:, :, -1, :]) #배치,윈도우,d_model
        #print('out_in_deoutputs', out_in_deoutputs.shape)
        
        # out - key decoder
        out_key_deinputs = [out_enoutputs, tkey_enoutputs]
        #print('out key deinputs', key_enoutputs.shape)
        out_key_deoutputs = self.out_key_dernn(out_key_deinputs, init_states) #enoutputs_[:, :, -1, :])
        #배치,윈도우,d_model
        #print('out_key_deoutputs', out_key_deoutputs.shape)
        
        # decoder concat
        out_key_in_outputs = self.out_key_in_concat(out_in_deoutputs, out_key_deoutputs) #배치,윈도우,d_model
        #print('out_key_in_outputs', out_key_in_outputs.shape)
        
        #out_key_in_outputs = tf.concat([out_in_deoutputs, out_key_deoutputs], axis = -1)
        #print(out_in_deoutputs.shape, out_key_deoutputs.shape, out_key_in_outputs.shape)

        # out logits
        out_key_in_outputs_ = self.out_key_in_deodense(out_key_in_outputs) #배치, 윈도우, out dim
        #print('out_key_in_outputs_', out_key_in_outputs_.shape)


        return key_in_outputs_, out_key_in_outputs_

