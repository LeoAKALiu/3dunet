from model import unet
from generator import *
from preprocess import *
from keras.callbacks import EarlyStopping
import keras.backend as K
import tensorflow as tf
import os


def train(image_path, mask_path):
    print('load data>>>>')
    image_train, image_valid, mask_train, mask_valid = preprocess_data_train(
        image_path, mask_path, size=64, replica=3, split=True)

    print('data loading complete!')

    print('model loaded>>>>')
    print('fitting model>>>>')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=tf.get_default_graph(), config=config) as sess:
        K.set_session(sess)
        sess.run(tf.global_variables_initializer())
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
        stop = EarlyStopping(patience=4)

        # checkpoint = ModelCheckpoint(filepath='/checkpoint-{epoch:02d}-{val_loss:.4f}.hdf5',
        #                             monitor='val_loss', verbose=1, save_best_only=True)


        model = unet(lr=1e-4)
        model.summary()
        model.fit_generator(generator=generator(image_train, mask_train),
                            steps_per_epoch=len(image_train),
                            epochs=10,
                            validation_data=[image_valid, mask_valid],
                            #validation_steps=64,
                            verbose=1,
                            callbacks=[stop])
        model.save_weights('./weight.h5')

if __name__ == '__main__':
    train(image_path='../image.npy', mask_path='../mask.npy')