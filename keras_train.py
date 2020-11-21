'''

keras: CNN Transcription model

'''
#from __future__ import print_function
import argparse

import matplotlib.pyplot as plt

#keras utils
from keras.callbacks import Callback
from keras import metrics
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D, add
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.optimizers import SGD
from keras import backend as K
from keras.utils import plot_model


import tensorflow as tf
import sklearn
from sklearn.metrics import precision_recall_fscore_support

#internal utils
from preprocess import DataGen
from config import load_config

import numpy as np

import os


def opt_thresholds(y_true,y_scores):
    othresholds = np.zeros(y_scores.shape[1])
    print("opt_thresholds() {}".format(othresholds.shape))
    for label, (label_scores, true_bin) in enumerate(zip(y_scores.T,y_true.T)):
        #print label
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(true_bin, label_scores)
        max_f1 = 0
        max_f1_threshold = .5
        for r, p, t in zip(recall, precision, thresholds):
            if p + r == 0: continue
            if (2*p*r)/(p + r) > max_f1:
                max_f1 = (2*p*r)/(p + r)
                max_f1_threshold = t
        #print label, ": ", max_f1_threshold, "=>", max_f1
        othresholds[label] = max_f1_threshold
        print("opt_thresholds() {}".format(othresholds))
    return othresholds

class linear_decay(Callback):
    '''
        decay = decay value to subtract each epoch
    '''
    def __init__(self, initial_lr,epochs):
        super(linear_decay, self).__init__()
        self.initial_lr = initial_lr
        self.decay = initial_lr/epochs

    def on_epoch_begin(self, epoch, logs={}):
        new_lr = self.initial_lr - self.decay*epoch
        print("ld: learning rate is now "+str(new_lr))
        K.set_value(self.model.optimizer.lr, new_lr)

class half_decay(Callback):
    '''
        decay = decay value to subtract each epoch
    '''
    def __init__(self, initial_lr,period):
        super(half_decay, self).__init__()
        self.init_lr = initial_lr
        self.period = period

    def on_epoch_begin(self, epoch, logs={}):
        factor = epoch // self.period
        lr  = self.init_lr / (2**factor)
        print("hd: learning rate is now "+str(lr))
        K.set_value(self.model.optimizer.lr, lr)

class Threshold(Callback):
    '''
        decay = decay value to subtract each epoch
    '''
    def __init__(self, val_data):
        super(Threshold, self).__init__()
        self.val_data = val_data
        _,y = val_data
        self.othresholds = np.full(y.shape[1],0.5)

    def on_epoch_end(self, epoch, logs={}):
        #find optimal thresholds on validation data
        x,y_true = self.val_data
        y_scores = self.model.predict(x)
        self.othresholds = opt_thresholds(y_true,y_scores)
        y_pred = y_scores > self.othresholds
        p,r,f,s = sklearn.metrics.precision_recall_fscore_support(y_true,y_pred,average='micro')
        print("validation p,r,f,s:")
        print(p,r,f,s)

def baseline_model():
    inputs = Input(shape=input_shape)
    reshape = Reshape(input_shape_channels)(inputs)

    #normal convnet layer (have to do one initially to get 64 channels)
    conv1 = Conv2D(50,(5,25),activation='tanh')(reshape)
    do1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling2D(pool_size=(1,3))(do1)

    conv2 = Conv2D(50,(3,5),activation='tanh')(pool1)
    do2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling2D(pool_size=(1,3))(do2)

    flattened = Flatten()(pool2)
    fc1 = Dense(1000, activation='sigmoid')(flattened)
    do3 = Dropout(0.5)(fc1)

    fc2 = Dense(200, activation='sigmoid')(do3)
    do4 = Dropout(0.5)(fc2)
    outputs = Dense(note_range, activation='sigmoid')(do4)

    model = Model(inputs=inputs, outputs=outputs)
    return model



def resnet_model(bin_multiple):

    #input and reshape
    inputs = Input(shape=input_shape)
    reshape = Reshape(input_shape_channels)(inputs)

    #normal convnet layer (have to do one initially to get 64 channels)
    conv = Conv2D(64,(1,bin_multiple*note_range),padding="same",activation='relu')(reshape)
    pool = MaxPooling2D(pool_size=(1,2))(conv)

    for i in range(int(np.log2(bin_multiple))-1):
        print("resnet_model() {}".format(i))
        #residual block
        bn = BatchNormalization()(pool)
        re = Activation('relu')(bn)
        freq_range = (bin_multiple/(2**(i+1)))*note_range
        print("resnet_model() {}".format(freq_range))
        conv = Conv2D(64,(1,int(freq_range)),padding="same",activation='relu')(re)

        #add and downsample
        ad = add([pool,conv])
        pool = MaxPooling2D(pool_size=(1,2))(ad)

    flattened = Flatten()(pool)
    fc = Dense(1024, activation='relu')(flattened)
    do = Dropout(0.5)(fc)
    fc = Dense(512, activation='relu')(do)
    do = Dropout(0.5)(fc)
    outputs = Dense(note_range, activation='sigmoid')(do)

    model = Model(inputs=inputs, outputs=outputs)

    return model

window_size = 7
min_midi = 21
max_midi = 108
note_range = max_midi - min_midi + 1


def train(args):
    path = os.path.join('models',args['model_name'])
    config = load_config(os.path.join(path,'config.json'))

    global feature_bins
    global input_shape
    global input_shape_channels

    bin_multiple = int(args['bin_multiple'])
    print('bin multiple',str(np.log2(bin_multiple)))
    feature_bins = note_range * bin_multiple
    input_shape = (window_size,feature_bins)
    input_shape_channels = (window_size,feature_bins,1)

    #filenames
    model_ckpt = os.path.join(path,'ckpt.h5')

    #train params
    batch_size = 256
    epochs = 1000

    trainGen = DataGen(os.path.join(path,'data','train'),batch_size,args)
    valGen = DataGen(os.path.join(path,'data','val'),batch_size,args)
    #valData = load_data(os.path.join(path,'data','val'))


    if os.path.isfile(model_ckpt):
        print('loading model')
        model = load_model(model_ckpt)
    else:
        print('training new model from scratch with bin multiple {0}'.format(bin_multiple))
        if bool(args['residual']):
            model = resnet_model(int(bin_multiple))
        else:
            model = baseline_model()

    init_lr = float(args['init_lr'])

    model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=init_lr,momentum=0.9))
    model.summary()
    plot_model(model, to_file=os.path.join(path,'model.png'))

    checkpoint = ModelCheckpoint(model_ckpt, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stop = EarlyStopping(patience=5,monitor='val_loss', verbose=1, mode='min')
    #tensorboard = TensorBoard(log_dir='./logs/baseline/', histogram_freq=250, batch_size=batch_size)
    if args['lr_decay'] == 'linear':
        decay = linear_decay(init_lr,epochs)
    else:
        decay = half_decay(init_lr,5)
    csv_logger = CSVLogger(os.path.join(path,'training.log'))
    #t = Threshold(valData)
    callbacks = [checkpoint,early_stop,decay,csv_logger]

    history = model.fit_generator(next(trainGen),trainGen.steps(), epochs=epochs,
              verbose=1,validation_data=next(valGen),validation_steps=valGen.steps(),callbacks=callbacks)

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    '''plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('baseline/acc.png')'''

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('baseline/loss.png')

    #test
    testGen = DataGen(os.path.join(path,'data','test'),batch_size,args)

    res = model.evaluate_generator(next(testGen),steps=testGen.steps())
    print(model.metrics_names)
    print(res)

def main():
    #train
    parser = argparse.ArgumentParser(
        description='Preprocess MIDI/Audio file pairs into ingestible data')
    parser.add_argument('model_name',
                        help='Path to the model directory where data should reside')

    args = vars(parser.parse_args())
    train(args)


if __name__ == '__main__':
    main()
