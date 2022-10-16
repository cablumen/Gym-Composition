import numpy as np
import random

import Settings


class DataManager:
    def __init__(self):
        # {action_index -> ([[state, predicted_next_state, next_state, reward, done], ...])
        self.__action_data = {}
        self.reset()
    
    def is_training_ready(self, action_index):
        return self.get_action_datasize(action_index) >= Settings.BATCH_SIZE * 2

    #       action data
    def put_action_data(self, action_index, action_record):
        return self.__action_data[action_index].append(action_record)
        
    def get_action_datasize(self, action_index):
        return len(self.__action_data[action_index])

    def get_action_data(self, action_index):
        return self.__get_random_action_data(action_index)

    #       critic data
    def get_critic_data(self):
        return self.__get_random_critic_data()

    def is_critic_training_ready(self):
        is_ready = True
        action_batchsize = int(Settings.BATCH_SIZE / Settings.ACTION_SIZE)
        for action_index in range(Settings.ACTION_SIZE):
            is_ready = is_ready and (self.get_action_datasize(action_index) >= action_batchsize)

        return is_ready

    def reset(self):
        for action_index in range(Settings.ACTION_SIZE):
            self.__action_data[action_index] = []

    #       private functions
    def __get_random_action_data(self, data_index):
        train_x = np.empty((Settings.BATCH_SIZE, Settings.OBSERVATION_SIZE), dtype=np.float32)
        train_y = np.empty((Settings.BATCH_SIZE, Settings.OBSERVATION_SIZE), dtype=np.float32)
        test_x = np.empty((Settings.BATCH_SIZE, Settings.OBSERVATION_SIZE), dtype=np.float32)
        test_y = np.empty((Settings.BATCH_SIZE, Settings.OBSERVATION_SIZE), dtype=np.float32)

        # randomly generate action indicies to sample
        random_indicies = random.sample(range(0, self.get_action_datasize(data_index)), Settings.BATCH_SIZE * 2)
        sampled_training_indicies = random_indicies[0:Settings.BATCH_SIZE]
        sampled_validation_indicies = random_indicies[Settings.BATCH_SIZE:]

        # populate test sample
        batch_index = 0
        for random_index in sampled_training_indicies:
            action_data = self.__action_data[data_index][random_index]
            train_x[batch_index] = action_data[0]
            train_y[batch_index] = action_data[2] - action_data[0]
            batch_index += 1

        
        # populate validation sample
        batch_index = 0
        for random_index in sampled_validation_indicies:
            action_data = self.__action_data[data_index][random_index]
            test_x[batch_index] = action_data[0]
            test_y[batch_index] = action_data[2] - action_data[0]
            batch_index += 1

        return train_x, train_y, test_x, test_y

    #       private functions
    def __get_random_critic_data(self):
        states = np.empty((Settings.BATCH_SIZE, Settings.OBSERVATION_SIZE), dtype=np.float32)
        next_states = np.empty((Settings.BATCH_SIZE, Settings.OBSERVATION_SIZE), dtype=np.float32)
        rewards = np.empty(Settings.BATCH_SIZE, dtype=float)
        dones = np.empty(Settings.BATCH_SIZE, dtype=np.int32)
    
        batch_index = 0
        action_sample_size = int(Settings.BATCH_SIZE / Settings.ACTION_SIZE)
        for action_index in range(Settings.ACTION_SIZE):
            # randomly generate action indicies to sample
            random_indicies = random.sample(range(0, self.get_action_datasize(action_index)), action_sample_size)

            # populate test sample
            for random_index in random_indicies:
                action_data = self.__action_data[action_index][random_index]

                states[batch_index] = action_data[0]
                next_states[batch_index] = action_data[2]
                rewards[batch_index] = action_data[3]
                dones[batch_index] = action_data[4]
                batch_index += 1

        return states, next_states, rewards, dones