from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pandas as pd
import numpy
import cv2

#Initial hyperparameters
#Here I will only be using the mass cc images
hold_path = '/Users/christopherlawton/final_test_train_hold/hold'

train_path = '/Users/christopherlawton/final_test_train_hold/train'
test_path = '/Users/christopherlawton/final_test_train_hold/test'


batch_size = 60
nb_classes = 2
nb_epoch = 5
kernel_size = (3, 3)
pool_size = (2,2)
strides = 1
input_shape = (150, 150, 1)

call_backs = [ModelCheckpoint(filepath='/Users/christopherlawton/galvanize/module_2/capstone_2/save_model/final_mod.h5',
                            monitor='val_loss',
                            save_best_only=True),
                            EarlyStopping(monitor='val_loss', patience=5, verbose=0)]

model = Sequential()

model.add(Convolution2D(32, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape,
                        kernel_initializer='glorot_uniform'))
# model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size, strides=strides))


model.add(Convolution2D(64, kernel_size[0], kernel_size[1],
                        kernel_initializer='glorot_uniform'))
# model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size, strides=strides))


model.add(Convolution2D(64, kernel_size[0], kernel_size[1],
                        kernel_initializer='glorot_uniform'))
# model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size, strides=strides))

# transition to an mlp
model.add(Flatten())
model.add(Dense(128, kernel_initializer='glorot_uniform'))
model.add(Activation('relu'))

model.add(Dense(64, kernel_initializer='glorot_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
                rescale=1./65535,
                rotation_range=0.4,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True)

test_datagen = ImageDataGenerator(
                rescale=1./65535)

train_generator = train_datagen.flow_from_directory(
                    train_path,
                    color_mode='grayscale',
                    target_size=(150,150),
                    batch_size=batch_size,
                    class_mode='binary',
                    shuffle=True)

validation_generator = test_datagen.flow_from_directory(
                    test_path,
                    color_mode='grayscale',
                    target_size=(150,150),
                    batch_size=batch_size,
                    class_mode='binary',
                    shuffle=True)

holdout_generator = test_datagen.flow_from_directory(
                    hold_path,
                    color_mode='grayscale',
                    target_size=(150,150),
                    batch_size=batch_size,
                    class_mode='binary',
                    shuffle=True)



history = model.fit_generator(
        train_generator,
        steps_per_epoch=520 // batch_size,
        epochs=nb_epoch,
        validation_data=validation_generator,
        callbacks=call_backs,
        validation_steps=148 // batch_size)



'''evaluation on hold out folder'''

holdout_generator = test_datagen.flow_from_directory(
                    '/Users/christopherlawton/final_test_train_hold/hold',
                    color_mode='grayscale',
                    target_size=(150,150),
                    batch_size=batch_size,
                    class_mode='binary',
                    shuffle=True)

metrics = model.evaluate_generator(holdout_generator,
                                    steps=74//batch_size,
                                    use_multiprocessing=True,
                                    verbose=1)

print(f"holdout loss: {metrics[0]} accuracy: {metrics[1]}")
#holdout loss: 0.6953471302986145 accuracy: 0.5 for final model

def plot_model(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) +1)

    plt.plot(epochs, acc, 'g-', label='Training acc')
    plt.plot(epochs, val_acc, 'b-', label='Validation acc')
    plt.title('Training and validation accuracy', fontsize=18)
    plt.xlabel('epochs', fontsize=16)
    plt.ylabel('loss', fontsize=16)
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'g-', label='Training loss')
    plt.plot(epochs, val_loss, 'b-', label='Validation loss')
    plt.title('Training and validation loss', fontsize=18)
    plt.xlabel('epochs', fontsize=16)
    plt.ylabel('loss', fontsize=16)
    plt.legend()
    plt.show()

plot_model(history)
