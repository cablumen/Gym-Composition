import gym
import numpy as np
import os
import time

from ActionDictionary import ActionDictionary
from Architectures import Architectures
import Settings

class Controller:
    def __init__(self):
        # self.__env = gym.make("CartPole-v1", render_mode='human')
        self.__env = gym.make("CartPole-v1")

        run_folder = self.get_run_folder()
        experiments = [(Architectures.FC_1_16, Architectures.REWARD_1_32)] # format: (sub_model_architecture, reward_architecture)

        self.run_experiments(run_folder, experiments)

    def get_run_folder(self):
        # get current directory path
        dir_path = os.path.dirname(os.path.abspath(__file__))

        # create sub-folder for logs
        log_folder = os.path.join(dir_path, "logs")
        if not os.path.isdir(log_folder):
            os.mkdir(log_folder)

        # create sub-folder for specific run
        run_folder = os.path.join(log_folder, str(time.strftime("%Y-%m-%d_%H-%M-%S")))
        if not os.path.isdir(run_folder):
            os.mkdir(run_folder)

        return run_folder

    def run_experiments(self, log_folder, experiments):
        for sub_model_architecture, reward_architecture in experiments:
            action_dictionary = ActionDictionary(log_folder, sub_model_architecture, reward_architecture)

            for action_dictionary.logger.session in range(Settings.SESSION_COUNT):
                session_start = time.time()
                self.epsilon_exploration(action_dictionary)
                action_dictionary.reset()
                session_end = time.time()
                session_duration = (session_end - session_start) * 1000 * 60
                # ~ 4.5 minutes. 
                print("session duration: " + str(session_duration) + " s")

    # execute best predicted action with increasing probability
    def epsilon_exploration(self, action_dictionary):
        print("\nController(epsilon_exploration): submodel:" + action_dictionary.get_submodel_name() + " reward:" + action_dictionary.get_reward_model_name())
        epsilon = Settings.EPSILON_START

        rewards_list = []
        for action_dictionary.logger.episode in range(Settings.EPISODE_COUNT):
            state = self.__env.reset()
            state = np.reshape(state, [1, Settings.OBSERVATION_SIZE])
            reward_for_episode = 0
            for action_dictionary.logger.step in range(Settings.MAX_TRAINING_STEPS):
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

            action_dictionary.evaluate_models()
            rewards_list.append(reward_for_episode)
            last_rewards_mean = np.mean(rewards_list[-30:])
            print("\tEpisode: ", action_dictionary.logger.episode, " || Reward: ", reward_for_episode, " || Average Reward: ", last_rewards_mean)
        
        return rewards_list


if __name__ == '__main__':
    Controller()