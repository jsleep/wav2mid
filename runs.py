import csv
from config import create_config
from preprocess import preprocess, organize
import multiprocessing
import argparse

def jobs(args):
    from keras_train import train

    print("Working on {}...".format(args['model_name']))
    create_config(args)
    preprocess(args)
    organize(args)
    train(args)

def main(pool):
    with open('models.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        pool.map(jobs, reader)
        return

if __name__== '__main__':
    # semaphore = multiprocessing.Semaphore(1)
    counts=multiprocessing.cpu_count()
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--threads', metavar='N',type=int, required=False, help='Numbers of threads for models (default=%(default)s, max={})'.format(str(counts)),default=1)

    args = vars(parser.parse_args())

    pool=multiprocessing.Pool(args['threads'])
    main(pool)