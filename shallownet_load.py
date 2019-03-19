import warnings
warnings.simplefilter('ignore
')
from hvdev.datasets import SimpleDatasetLoader
from hvdev.preprocessing import SimplePreprocessor
from hvdev.preprocessing import ImageToArrayPreprocessor
import matplotlib.pyplot as plt 
from keras.models import load_model
from imutils import paths
import numpy as np 
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required = True , help = 'path to the dataset')
ap.add_argument('-m', '--model', required = True , help = 'path to model saved')

args = vars(ap.parse_args())

classesLabels = ['cat', 'dog', 'panda']

print('[INFO] sampling images...')

imagePaths = np.array(list(paths.list_images(args['dataset'])))
idxs = np.random.randint(0, len(imagePaths), size = (10, ))
imagePaths = imagePaths[idxs]

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader([sp , iap])

(data , labels ) = sdl.load(imagePaths)

data = data.astype('float32')/255.0

print('[INFO] Loading pre treining network...')
model = load_model(args['model'])

print('[INFO] predicting...')
preds = model.predict(data, batch_size = 32).argmax(axis = 1)

for (i , imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    cv2.putText(image, "Label: {}".format(classesLabels[preds[i]]),
        (10 , 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Predictions',image)
    cv2.waitKey(0)