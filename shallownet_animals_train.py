import warnings
warnings.simplefilter('ignore')
from hvdev.datasets import SimpleDatasetLoader
from hvdev.preprocessing import SimplePreprocessor
from hvdev.preprocessing import ImageToArrayPreprocessor
from hvdev.nn.conv import ShallowNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.optimizers import SGD
import numpy as np 
import matplotlib.pyplot as plt 
from imutils import paths
import argparse
 
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required = True , help = "path to the dataset")
ap.add_argument('-o', '--output', required = True , help = "path to plot")
ap.add_argument('-m', '--model', required = True , help = 'path to save models')

args = vars(ap.parse_args())

sp = SimplePreprocessor(32 , 32)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader([sp , iap])

print('[INFO] loading data...')
imagePaths = list(paths.list_images(args['dataset']))

(data , labels) = sdl.load(imagePaths , verbose = 500)

data = data.astype('float32')/255.0

(trainX, testX , trainY , testY) = train_test_split(data , labels , test_size = 0.25 , 
    random_state = 42)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

classesNames = ['cat', 'dog', 'panda']

model = ShallowNet().build(height = 32 , width = 32, depth = 3 , classes = 3)

print('[INFO] compile model...')
model.compile(loss = 'categorical_crossentropy', optimizer = SGD(lr = 0.005),
    metrics = ['accuracy'])

print('[INFO] training model...')
H = model.fit(trainX , trainY, validation_data = (testX, testY), epochs = 100 , batch_size = 32 ,
    verbose = 1)

print('[INFO] Serializing model...')

model.save(args['model'])

print('[INFO] Evaluating model...')
predictions = model.predict(testX , batch_size = 32)

print(classification_report(testY.argmax(axis = 1), predictions.argmax(axis = 1), 
    target_names = classesNames))

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 100), H.history['loss'], label = 'train_loss')
plt.plot(np.arange(0, 100), H.history['val_loss'], label = 'val_loss')
plt.plot(np.arange(0, 100), H.history['acc'], label = 'train_acc')
plt.plot(np.arange(0, 100), H.history['val_acc'], label = 'val_acc')
plt.title('Training Accuracy/Loss')
plt.xlabel('#epochs')
plt.ylabel('losses')
plt.legend()
plt.savefig(args['output'])