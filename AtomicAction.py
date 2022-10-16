import numpy as np
import tensorflow as tf
import uuid

import Settings
from Settings import LogLevel

class AtomicAction:
    def __init__(self, logger, data_manager, architecture, action_index):
        self.__logger = logger
        self.__data_manager = data_manager
        self.__name = architecture.name + "_" + str(action_index)
        self.__id = uuid.uuid4()
        self.__action_index = action_index

        logger.add_model_writer(self.__name, self.__id)

        self.__architecture = architecture.value
        self.__model = None
        self.reset()

        self.__model_verbosity = True if Settings.LogLevel == 0 else False

    def get_name(self):
        return self.__name

    def train(self):
        self.__logger.print("AtomicAction(train): " + self.__name, LogLevel.INFO)
        if self.__data_manager.is_training_ready(self.__action_index):
            train_x, train_y, test_x, test_y = self.__data_manager.get_action_data(self.__action_index)
            training_history = self.__model.fit(train_x, train_y, batch_size=Settings.BATCH_SIZE, epochs=Settings.EPOCHS, validation_data=(test_x, test_y), verbose=self.__model_verbosity)
            self.__logger.log_training(self.__id, training_history)
        
    def predict(self, state):
        self.__logger.print("AtomicAction(predict): " + self.__name, LogLevel.INFO)
        return self.__model.predict(state, verbose=self.__model_verbosity)

    def predict_batch(self, state_batch):
        self.__logger.print("AtomicAction(predict_batch): " + self.__name, LogLevel.INFO)
        return self.__model.predict(state_batch, batch_size=Settings.BATCH_SIZE, verbose=self.__model_verbosity)

    def evaluate(self, episode):
        self.__logger.print("AtomicModel(evaluate): " + self.__name, LogLevel.INFO)
        if self.__data_manager.is_training_ready(self.__action_index):
            _, _, test_x, test_y = self.__data_manager.get_action_data(self.__action_index)
            predict_y = self.predict(test_x)
            mse = ((np.square(predict_y - test_y)).mean(axis=1)).mean()
            self.__logger.log_evaluation(episode, self.__id, mse)

    def reset(self):
        sub_model = tf.keras.models.clone_model(self.__architecture)
        sub_model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        self.__model = sub_model
