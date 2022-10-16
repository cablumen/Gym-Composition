import gym
import numpy as np

from ActionDictionary import ActionDictionary
from Architectures import Architectures
from Logger import Logger
from MetricCalculator import MetricCalculator
import Settings

class Controller:
    def __init__(self):
        self.__logger = Logger()
        # self.__env = gym.make("CartPole-v1", render_mode='human')
        self.__env = gym.make("CartPole-v1")
        # self.__metric_calculator  = MetricCalculator(self.__env)

        self.__experiments = [(Architectures.FC_1_16, Architectures.REWARD_1_32)] # format: (sub_model_architecture, reward_architecture)
        self.__action_dicts = []
        for sub_model_architecture, reward_architecture in self.__experiments:
            self.__action_dicts.append(ActionDictionary(sub_model_architecture, reward_architecture))

        for action_dict in self.__action_dicts:
            self.aggregate_epsilon_rewards(5, action_dict, 200)

    def aggregate_epsilon_rewards(self, session_count, action_dictionary, episode_count):
        eplison_rewards = []
        for session_index in range(session_count):
            eplison_rewards.append(self.epsilon_exploration(action_dictionary, episode_count))
            action_dictionary.reset()

        # TODO: add smoothing here
        reward_mean_by_episode_index = []
        for episode_index in range(episode_count):
            episode_index_rewards = np.array([i[episode_index] for i in eplison_rewards])
            episode_index_reward_mean = np.mean(episode_index_rewards)
            reward_mean_by_episode_index.append(episode_index_reward_mean)
        
        # TOOD: find better way determine filenames besides guids
        self.__logger.log_rewards(reward_mean_by_episode_index)

    # execute best predicted action with increasing probability
    def epsilon_exploration(self, action_dictionary, episode_count):
        print("\nController(epsilon_exploration): submodel:" + action_dictionary.get_submodel_name() + " reward:" + action_dictionary.get_reward_model_name())
        epsilon = Settings.EPSILON_START

        rewards_list = []
        for episode in range(episode_count):
            state = self.__env.reset()
            state = np.reshape(state, [1, Settings.OBSERVATION_SIZE])
            reward_for_episode = 0
            for step in range(Settings.MAX_TRAINING_STEPS):
                if np.random.rand() < epsilon:
                    action_index, predicted_next_state = action_dictionary.predict_random_action(state) 
                else:
                    action_index, predicted_next_state = action_dictionary.predict_actions(state) 

                next_state, reward, done, info = self.__env.step(action_index)
                action_dictionary.put_action_data(action_index, [state, predicted_next_state, next_state, reward, done])

                next_state = np.reshape(next_state, [1, Settings.OBSERVATION_SIZE])
                reward_for_episode += reward
                state = next_state

                action_dictionary.train_models()

                if done:
                    break
                
            # reduce probability of random action
            if epsilon > Settings.EPSILON_MIN:
                epsilon *= Settings.EPSILON_DECAY

            action_dictionary.evaluate_models(episode)
            rewards_list.append(reward_for_episode)
            last_rewards_mean = np.mean(rewards_list[-30:])
            print("\tEpisode: ", episode, " || Reward: ", reward_for_episode, " || Average Reward: ", last_rewards_mean)
        
        return rewards_list


if __name__ == '__main__':
    Controller()