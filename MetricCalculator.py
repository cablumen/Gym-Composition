import matplotlib.pyplot as plt
import numpy as np
import os
import time

import Settings


class MetricCalculator:
    def __init__(self, environment):
        self.__env = environment

        # get current directory path
        dir_path = os.path.dirname(os.path.abspath(__file__))
        
        # create sub-folder for all training data and graphs
        experiment_path = os.path.join(dir_path, "experiment records")
        if not os.path.isdir(experiment_path):
            os.mkdir(experiment_path)

        # create sub-folder for specific run
        self.__log_folder = os.path.join(experiment_path, str(time.strftime("%Y-%m-%d_%H-%M-%S")))
        if not os.path.isdir(self.__log_folder):
            os.mkdir(self.__log_folder)

    def calculate_divergence(self, action_dictionary, sequence_length):
        action_count = action_dictionary.get_action_count()
        for action_index in range(action_count):
            action_metric_averages = []
            while len(action_metric_averages) < 25:
                epsiode_metrics = []
                state = self.__env.reset()
                state = np.reshape(state, [1, Settings.OBSERVATION_SIZE])
                for step in range(Settings.MAX_TRAINING_STEPS):
                    step_action_index, predicted_next_state = action_dictionary.predict_action(action_index, state) 

                    next_state, reward, done, info = self.__env.step(step_action_index)                    
                    step_metric = self.__calculate_abs_dif(next_state, predicted_next_state)
                    epsiode_metrics.append(step_metric)

                    # every action_sequence_length steps, set the state to the real observed state
                    if step % sequence_length == sequence_length - 1:
                        state = np.reshape(next_state, [1, Settings.OBSERVATION_SIZE])

                    else:
                        state = np.reshape(predicted_next_state, [1, Settings.OBSERVATION_SIZE])

                    if done:
                        break
                
                # only sample episodes greater than the sequence length
                if len(epsiode_metrics) < sequence_length:
                    continue

                episode_metric_averages = []
                for action_sequence_index in range(sequence_length):
                    action_sequence_data = epsiode_metrics[action_sequence_index::sequence_length]
                    average_metrics = np.mean(action_sequence_data)
                    episode_metric_averages.append(average_metrics)

                action_metric_averages.append(episode_metric_averages)

            sequence_indices_metrics = []
            for sequence_index in range(sequence_length):
                sequence_index_metric = np.array([i[sequence_index] for i in action_metric_averages])
                sequence_index_metric_mean = np.mean(sequence_index_metric)
                sequence_indices_metrics.append(sequence_index_metric_mean)

            action_sequence_indices = range(1, sequence_length + 1)

            plt.figure()
            plt.xlabel("Action Sequence Index")
            plt.ylabel("Action Divergence (ABS DIF)")
            plt.plot(action_sequence_indices, sequence_indices_metrics)

            graph_path = os.path.join(self.__log_folder, "submodel_" + action_dictionary.get_submodel_name() + " reward_" + action_dictionary.get_reward_model_name() + " sequence_" + str(sequence_length) + " action_" + str(action_index) + ".png")
            plt.savefig(graph_path)
            plt.close()

    def __calculate_mse(self, observed_state, predicted_state):
        return np.square(np.subtract(observed_state, predicted_state)).mean()

    def __calculate_abs_dif(self, observed_state, predicted_state):
        return abs(np.subtract(observed_state, predicted_state)).mean()