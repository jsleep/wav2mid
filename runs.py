import csv
from config import create_config
from preprocess import preprocess, organize
from keras_train import train

with open('models.csv', 'rb') as csvfile:
     reader = csv.reader(csvfile)
     header = reader.next()
     #print header
     for row in reader:
         args = dict(zip(header,row))
         print args['model_name']
         create_config(args)
         preprocess(args)
         organize(args)
         train(args)
