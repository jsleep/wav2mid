import csv
from config import create_config
from preprocess import preprocess, organize
from keras_train import train

with open('models.csv', 'r') as csvfile:
     reader = csv.DictReader(csvfile)
     #print header
     for args in reader:
         print(args['model_name'])
         create_config(args)
         preprocess(args)
         organize(args)
         train(args)
