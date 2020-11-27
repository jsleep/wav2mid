
from collections import defaultdict
import sys, os
import argparse

import madmom
import numpy as np
import pandas as pd
import pretty_midi
import librosa
import h5py
import math

from config import load_config

import numpy as np

def readmm(d,args):
    ipath = os.path.join(d,'input.dat')
    note_range = 88
    n_bins = int(args['bin_multiple']) * note_range
    window_size = 7
    mmi = np.memmap(ipath, mode='r')
    i = np.reshape(mmi,(-1,window_size,n_bins))
    opath = os.path.join(d,'output.dat')
    mmo = np.memmap(opath, mode='r')
    o = np.reshape(mmo,(-1,note_range))
    return i,o

class DataGen:
    def __init__(self, dirpath, batch_size,args,num_files=1):
        print('initializing gen for '+dirpath)

        self.mmdirs =  os.listdir(dirpath)
        self.spe = 0 #steps per epoch
        self.dir = dirpath
        self.args = args

        for mmdir in self.mmdirs:
            print(mmdir)
            _,outputs = readmm(os.path.join(self.dir,mmdir),args)
            self.spe += len(outputs) // batch_size
            #print cnt
        self.num_files = num_files

        self.batch_size = batch_size
        self.current_file_idx = 0
        print('starting with ', self.mmdirs[self.current_file_idx:self.current_file_idx+self.num_files])
        for j in range(self.num_files):
            mmdir = os.path.join(self.dir,self.mmdirs[self.current_file_idx+j])
            print(mmdir)
            i,o = readmm(mmdir,args)
            if j == 0:
                self.inputs,self.outputs = i,o
                print('set inputs,outputs')
            else:
                self.inputs = np.concatenate((self.inputs,i))
                self.outputs = np.concatenate((self.outputs,o))
                print('concatenated')
            self.current_file_idx = (self.current_file_idx + 1) % len(self.mmdirs)
        self.i = 0

    def steps(self):
        return self.spe

    def __next__(self):
        while True:
            if (self.i+1)*self.batch_size > self.inputs.shape[0]:
                #return rest and then switch files
                x,y = self.inputs[self.i*self.batch_size:],self.outputs[self.i*self.batch_size:]
                self.i = 0
                if len(self.mmdirs) > 1: # no need to open any new files if we only deal with one, like for validation
                    print('switching to ', self.mmdirs[self.current_file_idx:self.current_file_idx+self.num_files])
                    for j in range(self.num_files):
                        mmdir = os.path.join(self.dir,self.mmdirs[self.current_file_idx+j])
                        i,o = readmm(mmdir,self.args)
                        if j == 0:
                            self.inputs,self.output = i,o
                        else:
                            self.inputs = np.concatenate((self.inputs,i))
                            self.outputs = np.concatenate((self.outputs,o))
                        self.current_file_idx = (self.current_file_idx + 1) % len(self.mmdirs)

            else:
                x,y = self.inputs[self.i*self.batch_size:(self.i+1)*self.batch_size],self.outputs[self.i*self.batch_size:(self.i+1)*self.batch_size]
                self.i += 1
            yield x,y

'''def load_data(dirpa):
    print('loading data from '+dirpath)
    hdf5_file = os.listdir(dirpath)[0]
    with h5py.File(os.path.join(dirpath,hdf5_file), 'r') as hf:
        inputs = hf['-inputs'][:]
        outputs = hf['-outputs'][:]
    return inputs,outputs'''


sr = 22050
hop_length = 512
window_size = 7
min_midi = 21
max_midi = 108


def wav2inputnp(audio_fn,spec_type='cqt',bin_multiple=3):
    print("wav2inputnp")
    bins_per_octave = 12 * bin_multiple #should be a multiple of 12
    n_bins = (max_midi - min_midi + 1) * bin_multiple

    #down-sample,mono-channel
    y,_ = librosa.load(audio_fn,sr)
    S = librosa.cqt(y,fmin=librosa.midi_to_hz(min_midi), sr=sr, hop_length=hop_length,
                      bins_per_octave=bins_per_octave, n_bins=n_bins)
    S = S.T

    #TODO: LogScaleSpectrogram?
    '''
    if spec_type == 'cqt':
        #down-sample,mono-channel
        y,_ = librosa.load(audio_fn,sr)
        S = librosa.cqt(y,fmin=librosa.midi_to_hz(min_midi), sr=sr, hop_length=hop_length,
                          bins_per_octave=bins_per_octave, n_bins=n_bins)
        S = S.T
    else:
        #down-sample,mono-channel
        y = madmom.audio.signal.Signal(audio_fn, sample_rate=sr, num_channels=1)
        S = madmom.audio.spectrogram.LogarithmicFilteredSpectrogram(y,fmin=librosa.midi_to_hz(min_midi),
                                            hop_size=hop_length, num_bands=bins_per_octave, fft_size=4096)'''

    #S = librosa.amplitude_to_db(S)
    S = np.abs(S)

    minDB = np.min(S)

    print(np.min(S),np.max(S),np.mean(S))

    S = np.pad(S, ((window_size//2,window_size//2),(0,0)), 'constant', constant_values=minDB)



    windows = []

    # IMPORTANT NOTE:
    # Since we pad the the spectrogram frame,
    # the onset frames are actually `offset` frames.
    # To obtain a window of the center frame at each true index, we take a slice from i to i+window_size
    # starting at frame 0 of the padded spectrogram
    for i in range(S.shape[0]-window_size+1):
        w = S[i:i+window_size,:]
        windows.append(w)


    #print inputs
    x = np.array(windows)
    return x

def mid2outputnp(pm_mid,times):
    piano_roll = pm_mid.get_piano_roll(fs=sr,times=times)[min_midi:max_midi+1].T
    piano_roll[piano_roll > 0] = 1
    return piano_roll



def joinAndCreate(basePath,new):
    newPath = os.path.join(basePath,new)
    if not os.path.exists(newPath):
        os.mkdir(newPath)
    return newPath

def isSplitFolder(ddir):
    return ddir == 'train' or ddir == 'test' or ddir == 'val'

def organize(args):
    valCnt = 1
    testPrefix = 'ENS'

    path = os.path.join('models',args['model_name'])
    dpath = os.path.join(path,'data')

    train_path = joinAndCreate(dpath,'train')
    test_path = joinAndCreate(dpath,'test')
    val_path = joinAndCreate(dpath,'val')

    for ddir in os.listdir(dpath):
        print("organize(): %s" % ddir)
        if os.path.isdir(os.path.join(dpath,ddir)) and not isSplitFolder(ddir):
            #print h5file
            if ddir.startswith(testPrefix):
                os.rename(os.path.join(dpath,ddir), os.path.join(test_path,ddir))
            elif valCnt > 0:
                os.rename(os.path.join(dpath,ddir), os.path.join(val_path,ddir))
                valCnt -= 1
            else:
                os.rename(os.path.join(dpath,ddir), os.path.join(train_path,ddir))

def modelDapathtaExists(path, s):
    dirs = ['val','test','train']
    for ddir in dirs:
        full_dir = os.path.join(path,'data',ddir,s)
        if os.path.isdir(full_dir):
            return True
    return False

def preprocess(args):
    #params
    path = os.path.join('models',args['model_name'])
    config = load_config(os.path.join(path,'config.json'))



    bin_multiple = int(args['bin_multiple'])
    spec_type = args['spec_type']




    framecnt = 0

    # hack to deal with high PPQ from MAPS
    # https://github.com/craffel/pretty-midi/issues/112
    pretty_midi.pretty_midi.MAX_TICK = 1e10

    for s in os.listdir(args['data_dir']):
        subdir = os.path.join(args['data_dir'],s)
        if not os.path.isdir(subdir):
            continue
        try:
            if not args.force:
                if modelDapathtaExists(path, s):
                    print("{} exists, skiping".format(s))
                    continue
        except AttributeError:
            if modelDapathtaExists(path, s):
                print("{} exists, skiping".format(s))
                continue

        # recursively search in subdir
        print(subdir)
        inputs,outputs = [],[]
        addCnt, errCnt = 0,0
        for dp, dn, filenames in os.walk(subdir):
            # in each level of the directory, look at filenames ending with .mid
            for f in filenames:
                # if there exists a .wav file and .midi file with the same name

                if f.endswith('.wav'):
                    audio_fn = f
                    fprefix = audio_fn.split('.wav')[0]
                    mid_fn = fprefix + '.mid'
                    txt_fn = fprefix + '.txt'
                    if mid_fn in filenames:
                        # wav2inputnp
                        audio_fn = os.path.join(dp,audio_fn)
                        # mid2outputnp
                        mid_fn = os.path.join(dp,mid_fn)

                        pm_mid = pretty_midi.PrettyMIDI(mid_fn)

                        inputnp = wav2inputnp(audio_fn,spec_type=spec_type,bin_multiple=bin_multiple)
                        times = librosa.frames_to_time(np.arange(inputnp.shape[0]),sr=sr,hop_length=hop_length)
                        outputnp = mid2outputnp(pm_mid,times)

                        # check that num onsets is equal
                        if inputnp.shape[0] == outputnp.shape[0]:
                            print("adding to dataset fprefix {}".format(fprefix))
                            addCnt += 1
                            framecnt += inputnp.shape[0]
                            print("framecnt is {}".format(framecnt))
                            inputs.append(inputnp)
                            outputs.append(outputnp)
                        else:
                            print("error for fprefix {}".format(fprefix))
                            errCnt += 1
                            print(inputnp.shape)
                            print(outputnp.shape)

        print("{} examples in dataset".format(addCnt))
        print("{} examples couldnt be processed".format(errCnt))


        # concatenate dynamic list to numpy list of example
        if addCnt:
            inputs = np.concatenate(inputs)
            outputs = np.concatenate(outputs)

            fn = subdir.split('/')[-1]
            if not fn:
                fn = subdir.split('/')[-2]
            #fn += '.h5'
            # save inputs,outputs to hdf5 file
            datapath = joinAndCreate(path,'data')
            fnpath = joinAndCreate(datapath,fn)

            mmi = np.memmap(filename=os.path.join(fnpath,'input.dat'), mode='w+',shape=inputs.shape)
            mmi[:] = inputs[:]
            mmo = np.memmap(filename=os.path.join(fnpath,'output.dat'), mode='w+',shape=outputs.shape)
            mmo[:] = outputs[:]
            del mmi
            del mmo

            '''with h5py.File(os.path.join(datapath,fn), 'w') as hf:
                hf.create_dataset("-inputs",  data=inputs)
                hf.create_dataset("-outputs",  data=outputs)

                without dB, i'm just going to not worry about feature scaling
                if args.zn:
                    nppath = os.path.join(path,'xn')
                    if os.path.isfile(nppath+'.npz'):
                        npzfile = np.load(nppath+'.npz')
                        x,x2,n = npzfile['x'],npzfile['x2'],npzfile['n']
                    else:
                        x,x2,n = 0,0,0


                    x += np.sum(inputs,axis=0)
                    x2 += np.sum(inputs**2,axis=0)
                    n += inputs.shape[0]

                    print x,x2,n

                    print 'mean={}'.format(x/n)

                    print 'var={}'.format(x2/n-(x/n)**2)

                    np.savez(nppath,x=x,x2=x2,n=n)'''


if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Preprocess MIDI/Audio file pairs into ingestible data')

    parser.add_argument('model_name',
                        help='model name. will use config from directory and save preprocessed data to it')

    parser.add_argument('data_dir', default='../maps/',
                        help='Path to data dir, searched recursively, used for naming HDF5 file (default: %(default)s)')

    parser.add_argument('-b', dest='bin_multiple', type=int, default=4,
                        help='bin multiple (default: %(default)s)')

    parser.add_argument('-s', dest='spec_type', default='cqt',
                        help='Spec type (default: %(default)s)')

    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='Force overwrite existed model data (default: %(default)s)')

    parser.add_argument('--no-zn', dest='zn', action='store_false')
    parser.set_defaults(zn=True)

    args = vars(parser.parse_args())

    preprocess(args)
