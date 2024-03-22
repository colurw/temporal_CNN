import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import concatenate
from keras.utils import plot_model
import pickle

# load data
with open('data/all_data.pkl', 'rb') as file:
    price_series_data, time_cat_data, target_cat_data = pickle.load(file) 

# test inputs
train_x = np.random.rand(1,40,5)
train_y = np.random.rand(1,3)

timesteps = train_x.shape[1]
channels = train_x.shape[2]

input_cnn = Input(shape=(40,5,))
input_mlp = Input(shape=(31,))

# convolutional branch
cnn = Conv1D(filters=64, padding='same', kernel_size=3, dilation_rate=1, 
                 activation='relu', data_format='channels_last',
                 kernel_regularizer=None, bias_regularizer=None,
                 input_shape=(timesteps, channels))(input_cnn)

cnn = Conv1D(filters=64, padding='same', kernel_size=3, dilation_rate=1, 
                 activation='relu', data_format='channels_last', 
                 kernel_regularizer=None, bias_regularizer=None)(cnn)

cnn = Dropout(0.0)(cnn)
cnn = MaxPooling1D(pool_size=2, padding='same')(cnn)
cnn = Flatten()(cnn)
cnn = Model(inputs=input_cnn, outputs=cnn)

# perceptron branch
mlp = Dense(8, activation='relu')(input_mlp)
mlp = Model(inputs=input_mlp, outputs=mlp)

# join branches
combined = concatenate([cnn.output, mlp.output])
head = Dense(256, activation='relu')(combined)
head = Dense(64, activation='relu')(head)
head = Dense(3, activation='softmax')(head)
model = Model(inputs=[cnn.input, mlp.input], outputs=head)

#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

epochs = 10
batch_size = 32

# fit network
# model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=False)
# evaluate model
# _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
# print(accuracy)


## pad - most recent edge is important!
## replace maxpooling with dilated kernel to preserve data - geoff hinton
## replace maxpooling with stride = 2 - 'learnable pooling, all-convolution network'
## batch norm on conv layers?
## groups = channels maintains depthwise separation.  Add same multiple of filters?
## tf.data.shuffle()
## l2 less sensitive to small changes (noise), l1 blocks some inputs
## layers.DepthwiseConv1D() reduces number of parameters