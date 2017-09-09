import os
import argparse
import json

'''
This script is for creating and loading a JSON structure that will hold parameters that are to be
held constant for preprocessing data for, training, and testing each model used.
'''


def load_config(json_fn):
    with open(json_fn, 'r') as infile:
        config = json.load(infile)
    return config

def create_config(args):
    path = os.path.join('models',args['model_name'])
    if not os.path.exists(path):
        os.mkdir(path)
    with open(os.path.join(path,'config.json'), 'w') as outfile:
        json.dump(args, outfile)


if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Create a config JSON')

    #possible types/values
    #model_name,spec_type,init_lr,lr_decay,bin_multiple,residual,filter_shape
    #baseline,cqt,1e-2,linear,36,False,some
    #new,logstft,1e-1,geo,96,True,full

    parser.add_argument('model_name',
                        help='model name. will create a directory for model where config,data,etc will go')
    parser.add_argument('spec_type',
                        help='Spectrogram Type, cqt or logstft')
    parser.add_argument('init_lr', type=float,
                        help='Initial Learning Rate')
    parser.add_argument('lr_decay',
                        help='How the Learning Rate Will Decay')
    parser.add_argument('bin_multiple', type=int,
                        help='Used to calculate bins_per_octave')
    parser.add_argument('residual', type=bool,
                        help='Use Residual Connections or not')
    parser.add_argument('full_window',
                        help='Whether or not the convolution window spans the full axis')

    ''' These are all constant.
    parser.add_argument('--sr', type=int, default=22050,
                        help='Sampling Rate')
    parser.add_argument('--hl', type=int, default=512,
                        help='Hop Length')
    parser.add_argument('--ws', type=int, default=7,
                        help='Window Size')
    parser.add_argument('--bm', type=int, default=3,
                        help='Bin Multiple')
    parser.add_argument('--min', type=int, default=21, #A0
                        help='Min MIDI value')
    parser.add_argument('--max', type=int, default=108, #C8
                        help='Max MIDI value')'''

    args = vars(parser.parse_args())

    create_config(args)
