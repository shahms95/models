import tensorflow as tf
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.vgg16 import VGG16
from keras import optimizers
from python_utils import *
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import matplotlib.pyplot as plt
import tkinter
import os

# os.remove("data.txt")

learning_rate_vector1 = [0.005, 0.01, 0.015, 0.02, 0.025]
learning_rate_vector2 = [0.03, 0.035, 0.04, 0.045, 0.05]
batch_size_vector = [8, 16, 24, 32, 40, 48, 56, 64]
momentum_vector = [0.5, 0.9, 0.99, 1.0]

fspace1 = {
    'learning_rate': hp.choice('learning_rate', learning_rate_vector1),
    'batch_size': hp.choice('batch_size', batch_size_vector),
    'momentum': hp.choice('momentum', momentum_vector)
}

fspace2 = {
    'learning_rate': hp.choice('learning_rate', learning_rate_vector2),
    'batch_size': hp.choice('batch_size', batch_size_vector),
    'momentum': hp.choice('momentum', momentum_vector)
}

def f(params):
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    momentum = params['momentum']
    # batch_size = 
    config = tf.ConfigProto( device_count = {'GPU': 1 } ) 
    sess = tf.Session(config=config) 
    K.set_session(sess)

    model = VGG16(include_top=True, weights=None)

    # print(model.summary())

    sgd = optimizers.SGD(lr=learning_rate, clipnorm=1., momentum=momentum)

    model.compile(sgd, loss='categorical_crossentropy')

    ROOT_DIR = '../imagenet-project/ILSVRC/Data/CLS-LOC/'


    train_datagen  = ImageDataGenerator()
    test_datagen = ImageDataGenerator()
        
    img_rows, img_cols = 224,224 # 299x299 for inception, 224x224 for VGG and Resnet
    train_generator = train_datagen.flow_from_directory(
            ROOT_DIR + 'train/',
            target_size=(img_rows, img_cols),#The target_size is the size of your input images,every image will be resized to this size
            batch_size=batch_size,
            class_mode='categorical')

    print("Train Generator's work is done!")

    validation_generator = test_datagen.flow_from_directory(
            ROOT_DIR + 'val/',
            target_size=(img_rows, img_cols),#The target_size is the size of your input images,every image will be resized to this size
            batch_size=batch_size,
            class_mode='categorical')



    print("Validation Generator's work is done!")

    model.fit_generator(
            train_generator,
            steps_per_epoch=10,
            epochs=5, validation_data=validation_generator,
            validation_steps=50
            )
    # res = model.predict(x = validation_generator.next()[0])

    # print(res)

    res = model.evaluate(x = train_generator.next()[0], y = train_generator.next()[1], steps = 10)


    return {'loss': res, 'status': STATUS_OK}


trials = Trials()
best = fmin(fn=f, space=fspace1, algo=tpe.suggest, max_evals=20, trials=trials)

# print('best:', best)

# print('trials:')
# for trial in trials.trials[:2]:
#     print(trial)

# print(len(trials.trials))



# f, ax = plt.subplots(1)
learning_rate_s = [t['misc']['vals']['learning_rate'] for t in trials.trials]
batch_size_s = [t['misc']['vals']['batch_size'] for t in trials.trials]
momentums = [t['misc']['vals']['momentum'] for t in trials.trials]
losss = [t['result']['loss'] for t in trials.trials]



with open('vgg_data.txt', 'w+') as f:
    for i1, i2, i3, i4 in zip(learning_rate_s, batch_size_s, losss, momentums):
        f.write(str(learning_rate_vector1[i1[0]]))
        f.write(', ')
        f.write(str(batch_size_vector[i2[0]]))
        f.write(', ')
        f.write(str(momentum_vector[i4[0]]))
        f.write(', ')
        f.write(str(i3))
        f.write('\n')

best = fmin(fn=f, space=fspace2, algo=tpe.suggest, max_evals=20, trials=trials)

learning_rate_s = [t['misc']['vals']['learning_rate'] for t in trials.trials]
batch_size_s = [t['misc']['vals']['batch_size'] for t in trials.trials]
momentums = [t['misc']['vals']['momentum'] for t in trials.trials]
losss = [t['result']['loss'] for t in trials.trials]



with open('vgg_data.txt', 'a+') as f:
    for i1, i2, i3, i4 in zip(learning_rate_s, batch_size_s, losss, momentums):
        f.write(str(learning_rate_vector2[i1[0]]))
        f.write(', ')
        f.write(str(batch_size_vector[i2[0]]))
        f.write(', ')
        f.write(str(momentum_vector[i4[0]]))
        f.write(', ')
        f.write(str(i3))
        f.write('\n')

