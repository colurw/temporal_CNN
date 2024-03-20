import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
# from keras.utils import to_categorical
from keras.utils import plot_model

train_x = np.random.rand(1,40,5)
train_y = np.random.rand(1,3)

timesteps = train_x.shape[1]
channels = train_x.shape[2]

model = Sequential()
model.add(Conv1D(filters=65, groups=1, kernel_size=3, dilation_rate=1, 
                 activation='relu', 
                 kernel_regularizer=None, bias_regularizer=None,
                 input_shape=(timesteps, channels)))

model.add(Conv1D(filters=65, groups=1, kernel_size=3, dilation_rate=1, 
                 activation='relu', 
                 kernel_regularizer=None, bias_regularizer=None))

# model.add(Dropout(0.5))
# model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

epochs = 10
batch_size = 32

# fit network
model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=False)
# evaluate model
# _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
# print(accuracy)


## pad - most recent edge is important!
## replace maxpooling with dilated kernel to preserve data - geoff hinton.
## replace maxpooling with stride = 2 - 'learnable pooling, all-convolution network'
## batch norm on conv layers?
## groups = channels maintains depthwise separation.  Add same multiple of filters?
## normalise by delta from SMA
## tf.data.shuffle()
## l2 less sensitive to small changes (noise), l1 blocks some inputs
## layers.DepthwiseConv1D() reduces number of parameters