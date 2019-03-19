from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation , Dense , Flatten
from keras import backend as K 

class ShallowNet:
    @staticmethod
    def build(height , width , depth , classes):
        inputShape = (height , width , depth)

        if K.image_data_format() == 'channels_first':
            inputShape = (depth , height , width)

        model = Sequential()
        model.add(Conv2D(32 , (3, 3), padding = 'same', strides = (1, 1),
            input_shape = inputShape))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model