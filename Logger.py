import csv
import matplotlib.pyplot as plt
import os
import time
import uuid

import Settings

class Logger:
    def __init__(self, logging_enabled=True):
        self.__logging_enabled = logging_enabled

        self.__model_train_writers = {}
        self.__model_evaluate_writers = {}

        # get current directory path
        dir_path = os.path.dirname(os.path.abspath(__file__))
        
        # create sub-folder for all training data and graphs
        training_path = os.path.join(dir_path, "training records")
        if not os.path.isdir(training_path):
            os.mkdir(training_path)

        # create sub-folder for specific run
        self.__log_folder = os.path.join(training_path, str(time.strftime("%Y-%m-%d_%H-%M-%S")))
        if not os.path.isdir(self.__log_folder):
            os.mkdir(self.__log_folder)

        self.__set_plt_params()

    def print(self, string, log_level = 0):
        if self.__logging_enabled and log_level.value >= Settings.LOG_LEVEL:
            print(string)

    def add_model_writer(self, model_name, model_id):
        # create csv writer
        training_path = os.path.join(self.__log_folder, model_name + '_training.csv')
        evaluation_path = os.path.join(self.__log_folder, model_name + '_evaluation.csv')

        # find a unique filepath for the writers
        file_index = 1
        while(os.path.exists(training_path) or os.path.exists(evaluation_path)):
            training_path = os.path.join(self.__log_folder, model_name + '_training_' + str(file_index) + '.csv')
            evaluation_path = os.path.join(self.__log_folder, model_name + '_evaluation_' + str(file_index) + '.csv')
            file_index += 1

        #       create training file writer
        training_file = open(training_path, 'w', newline='')
        self.__model_train_writers[model_id] = csv.writer(training_file)

        # write column headers
        self.__model_train_writers[model_id].writerow(["Epoch", "Training MSE", "Validation MSE"])
        

        #       create evaluation file writer
        evaluation_file = open(evaluation_path, 'w', newline='')
        self.__model_evaluate_writers[model_id] = csv.writer(evaluation_file)

        # write column headers
        self.__model_evaluate_writers[model_id].writerow(["Step", "Evaluation MSE"])

    def log_training(self, model_id, training_history):
        # write training history
        for epoch in range(len(training_history.epoch)):
            epoch_mse = training_history.history["mse"][epoch]
            epoch_val_mse = training_history.history["val_mse"][epoch]
            # TOOD: add offset for epoch
            self.__model_train_writers[model_id].writerow([epoch, epoch_mse, epoch_val_mse])

    def log_evaluation(self, episode, model_id, mse):
        self.__model_evaluate_writers[model_id].writerow([episode, mse])

    def log_rewards(self, epsilon_rewards):
        reward_path = os.path.join(self.__log_folder, str(uuid.uuid4()) + '_rewards.csv')
        reward_file = open(reward_path, 'w', newline='')
        reward_writer = csv.writer(reward_file)
        reward_writer.writerow(["Episode Index", "Average Reward"])

        for episode_index in range(len(epsilon_rewards)):
            episode_index_mean = epsilon_rewards[episode_index]
            reward_writer.writerow([episode_index, episode_index_mean])

    def visualize_training(self, model_name, training_history):
        # AtomicModel logs
        plt.figure()
        plt.title(model_name + " training history")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        epoch = training_history.epoch
        accuracy = training_history.history["accuracy"]
        val_accuracy = training_history.history["val_accuracy"]
        
        plt.plot(epoch, accuracy, "-r", label="Accuracy")
        plt.plot(epoch, val_accuracy, "-b", label="Validation Accuracy")

        plt.legend(loc="center right")
        fig_path = os.path.join(self.__log_folder, model_name + " training.png")
        plt.savefig(fig_path)

    #
    def __get_unique_path(self, path):
        file_index = 1
        while(os.path.exists(path)):
            training_path = os.path.join(self.__log_folder, model_name + '_training_' + str(file_index) + '.csv')
            file_index += 1
        return 

    def __set_plt_params(self):
        background_color = "#1E1E1E"
        foreground_color = "#DBDBDB"
        plt.rcParams["figure.facecolor"] = background_color
        plt.rcParams["figure.edgecolor"] = background_color

        plt.rcParams["axes.facecolor"] = background_color
        plt.rcParams["axes.edgecolor"] = foreground_color
        plt.rcParams["axes.titlecolor"] = foreground_color
        plt.rcParams["axes.labelcolor"] = foreground_color

        plt.rcParams["xtick.color"] = foreground_color
        plt.rcParams["xtick.labelcolor"] = foreground_color
        plt.rcParams["ytick.color"] = foreground_color
        plt.rcParams["ytick.labelcolor"] = foreground_color

        plt.rcParams["legend.labelcolor"] = foreground_color
