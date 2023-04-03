import numpy as np
import tensorflow as tf

import Settings
from Settings import LogLevel

class AtomicAction:
    def __init__(self, logger, data_manager, architecture, action_index):
        self.name = architecture.name + "_" + str(action_index)

        self.__logger = logger
        self.__data_manager = data_manager
        self.__action_index = action_index

        self.__architecture = architecture.value
        self.__model = None
        self.reset()

        self.__model_verbosity = True if Settings.LogLevel == 0 else False

    @tf.function
    def train(self):
        self.__logger.print("AtomicAction(train): " + self.name, LogLevel.INFO)
        if self.__data_manager.is_training_ready(self.__action_index):
            train_x, train_y, test_x, test_y = self.__data_manager.get_action_data(self.__action_index)
            training_history = self.__model.fit(train_x, train_y, batch_size=Settings.BATCH_SIZE, epochs=Settings.EPOCHS, validation_data=(test_x, test_y), verbose=self.__model_verbosity)
            self.__logger.log_training(self.__action_index, training_history)
        
    def predict(self, state):
        self.__logger.print("AtomicAction(predict): " + self.name, LogLevel.INFO)
        return self.__model.predict(state, verbose=self.__model_verbosity)

    def evaluate(self):
        self.__logger.print("AtomicModel(evaluate): " + self.name, LogLevel.INFO)
        if self.__data_manager.is_training_ready(self.__action_index):
            _, _, test_x, test_y = self.__data_manager.get_action_data(self.__action_index)
            # TODO: why not predict batch here?
            predict_y = self.predict(test_x)
            mse = ((np.square(predict_y - test_y)).mean(axis=1)).mean()
            self.__logger.log_evaluation(self.__action_index, mse)

    def reset(self):
        sub_model = tf.keras.models.clone_model(self.__architecture)
        sub_model.compile(optimizer='adam', loss='mse', metrics=['mse'], jit_compile=True)
        self.__model = sub_model
