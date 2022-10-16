import numpy as np
import random

from AtomicAction import AtomicAction
from DataManager import DataManager
from Logger import Logger
from ActionCritic import ActionCritic
import Settings


class ActionDictionary:
    def __init__(self, sub_model_architecture, critic_architecture):
        self.__logger = Logger()
        self.__data_manager = DataManager()

        self.__action_critic = ActionCritic(self.__logger, self.__data_manager, critic_architecture)

        self.__action_list = []
        for action_index in range(Settings.ACTION_SIZE):
            self.__action_list.append(AtomicAction(self.__logger, self.__data_manager, sub_model_architecture, action_index))

        self.__training_step = 0
    
    def get_reward_model_name(self):
        return self.__action_critic.get_name()

    def get_submodel_name(self):
        return self.__action_list[0].get_name()

    def get_action_count(self):
        return len(self.__action_list)

    def put_action_data(self, action_index, action_record):
        self.__data_manager.put_action_data(action_index, action_record)

    def predict_random_action(self, state):
        random_action_index = random.randint(0, len(self.__action_list) - 1)
        delta_next_state = self.__action_list[random_action_index].predict(state)
        predicted_next_state = state + delta_next_state
        return random_action_index, predicted_next_state

    def predict_action(self, action_index, state):
        delta_next_state = self.__action_list[action_index].predict(state)
        predicted_next_state = state + delta_next_state
        return action_index, predicted_next_state

    def predict_actions(self, state):
        predicted_next_states = []
        predicted_rewards = []
        for action in self.__action_list:
            delta_next_state = action.predict(state)
            predicted_next_state = state + delta_next_state
            predicted_next_states.append(predicted_next_state)
            predicted_rewards.append(self.__action_critic.predict(predicted_next_state))

        optimal_action_index = np.argmax(predicted_rewards)
        optimal_next_state = predicted_next_states[optimal_action_index]
        return optimal_action_index, optimal_next_state

    def train_models(self):
        if self.__training_step % 5 == 0:
            self.__training_step = 0
            
            # train critic model
            self.__action_critic.train()

            # train actor models
            for action in self.__action_list:
                action.train() 

        self.__training_step += 1

    def evaluate_models(self, episode):
        for action in self.__action_list:
            action.evaluate(episode)
    
    def reset(self):
        self.__data_manager.reset()
        self.__action_critic.reset()
        for action in self.__action_list:
            action.reset()

            