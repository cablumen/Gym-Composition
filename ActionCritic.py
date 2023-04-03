import numpy as np
import tensorflow as tf

import Settings
from Settings import LogLevel

import time

GAMMA = 0.95

class ActionCritic:
    def __init__(self, logger, data_manager, architecture):
        self.name = architecture.name

        self.__logger = logger
        self.__data_manager = data_manager

        self.__architecture = architecture.value
        self.__model = None
        self.reset()

        self.__model_verbosity = True if Settings.LogLevel == 0 else False

    @tf.function
    def train(self):
        self.__logger.print("ActionCritic(train): " + self.name, LogLevel.INFO)
        if self.__data_manager.is_critic_training_ready():
            states, next_states, rewards, dones = self.__data_manager.get_critic_data()
            targets = rewards + GAMMA * self.predict_batch(next_states) * (1 - dones)
            self.__model.fit(states, targets, epochs=Settings.EPOCHS, verbose=self.__model_verbosity)
        
    def predict(self, state):
        self.__logger.print("ActionCritic(predict): " + self.name, LogLevel.INFO)
        return self.__model.predict(state, verbose=self.__model_verbosity)

    def predict_batch(self, state_batch):
        self.__logger.print("ActionCritic(predict_batch): " + self.name, LogLevel.INFO)
        return self.__model.predict(state_batch, batch_size=Settings.BATCH_SIZE, verbose=self.__model_verbosity)

    def reset(self):
        critic_model = tf.keras.models.clone_model(self.__architecture)
        critic_model.compile(optimizer='adam', loss='mse', metrics=['mse'], jit_compile=True)
        self.__model = critic_model