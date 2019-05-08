import tensorflow as tf
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.vgg16 import VGG16
from keras import optimizers
import numpy as np
import os 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", type=int, default=0,
                    help="The ID of GPU to be used; default = 0")
args = parser.parse_args() 
config = tf.ConfigProto() 

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
print("CUDA visible devices : ", os.environ["CUDA_VISIBLE_DEVICES"])

config.gpu_options.allow_growth=True


sess = tf.Session(config=config) 
K.set_session(sess)

model = VGG16(include_top=True, weights=None)

# print(model.summary())

sgd = optimizers.SGD(lr=0.01, clipnorm=1.)

model.compile(sgd, loss='categorical_crossentropy', metrics=['accuracy'])

ROOT_DIR = '../imagenet-project/ILSVRC/Data/CLS-LOC/'


train_datagen  = ImageDataGenerator()
test_datagen = ImageDataGenerator()
    
img_rows, img_cols = 224,224 # 299x299 for inception, 224x224 for VGG and Resnet
train_generator = train_datagen.flow_from_directory(
        ROOT_DIR + 'train/',
        target_size=(img_rows, img_cols),#The target_size is the size of your input images,every image will be resized to this size
        batch_size=32,
        class_mode='categorical')

print("Train Generator's work is done!")

validation_generator = test_datagen.flow_from_directory(
        ROOT_DIR + 'val/',
        target_size=(img_rows, img_cols),#The target_size is the size of your input images,every image will be resized to this size
        batch_size=32,
        class_mode='categorical')

print("Validation Generator's work is done!")

history = model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=3, validation_data=validation_generator,
        validation_steps=50
        )

with open('vgg16_acc.txt', 'w+') as f:
    for i1, i2, i3 in zip(history.epoch, history.history['acc'], history.history['loss']):
        f.write(str(i1))
        f.write(', ')
        f.write(str(i2))
        f.write(', ')
        f.write(str(i3))
        f.write('\n')