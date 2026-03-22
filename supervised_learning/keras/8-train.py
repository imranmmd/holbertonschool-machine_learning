#!/usr/bin/env python3
"""Trains a model using mini-batch gradient descent"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent"""
    callbacks = []

    if validation_data is not None:
        if early_stopping:
            early_stop = K.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience
            )
            callbacks.append(early_stop)

        if learning_rate_decay:
            def lr_decay(epoch):
                return alpha / (1 + decay_rate * epoch)

            lr_scheduler = K.callbacks.LearningRateScheduler(
                lr_decay,
                verbose=1
            )
            callbacks.append(lr_scheduler)

        if save_best:
            checkpoint = K.callbacks.ModelCheckpoint(
                filepath=filepath,
                monitor='val_loss',
                save_best_only=True
            )
            callbacks.append(checkpoint)

    return network.fit(data, labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=callbacks)
