from keras.engine.topology import Layer
from keras import initializations
from keras import backend as K

class AttLayer(Layer):
    def __init__(self,**kwargs):
        self.init = initializations.get('normal')
        super(AttLayer,self).__init__(**kwargs)

    def build(self,input_shape):
        assert len(input_shape)==3

        self.W = self.init((input_shape[-1],))
        ''' self.W = self.add_weight(shape=(input_shape[-1],input_shape[-1]), initializer='random_uniform', trainable=True) '''
        self.trainable_weights = [self.W]
        super(AttLayer,self).build(input_shape)

    def call(self,x,mask = None):
        eij = K.tanh(K.dot(x,self.W))

        ai = K.exp(eij)
        weights = ai/K.sum(ai,axis = 1).dimshuffle(0,'x')

        weighted_input = x*weights.dimshuffle(0,1,'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self,input_shape):

        return (input_shape[0],input_shape[-1])