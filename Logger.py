import csv
import matplotlib.pyplot as plt
import os

import Settings

class Logger:
    def __init__(self, architecture_folder, logging_enabled=True):
        self.__architecture_folder = architecture_folder
        self.__logging_enabled = logging_enabled

        # create sub-folders for architecture data
        self.__session_path = None
        self.__episode_path = None

        self.__summary_folder = os.path.join(architecture_folder, "Summary")
        if not os.path.isdir(self.__summary_folder):
            os.mkdir(self.__summary_folder)
            
        self.__reward_writer = None

        # {action_index -> writer}
        self.__model_train_writers = {}
        self.__model_evaluate_writers = {}

        self.__set_plt_params()

    @property
    def session(self):
        return self._session

    @session.setter
    def session(self, value):
        self.__session_path = os.path.join(self.__architecture_folder, "Session " + str(value))
        if not os.path.isdir(self.__session_path):
            os.mkdir(self.__session_path)

        self._session = value

    @property
    def episode(self):
        return self._episode

    @episode.setter
    def episode(self, value):
        self.__episode_path = os.path.join(self.__session_path, "Episode " + str(value))
        if not os.path.isdir(self.__episode_path):
            os.mkdir(self.__episode_path)

        #       create reward file writer
        reward_path = os.path.join(self.__episode_path, "Reward")
        reward_file = open(reward_path, 'w', newline='')
        self.__reward_writer = csv.writer(reward_file)
        self.__reward_writer.writerow(["Step", "Reward"])
        
        #       create sub-model writers
        for action_index in range(Settings.ACTION_SIZE):
            # create training file writer
            submodel_training_path = os.path.join(self.__episode_path, str(action_index) + '_training.csv')
            training_file = open(submodel_training_path, 'w', newline='')
            self.__model_train_writers[action_index] = csv.writer(training_file)
            self.__model_train_writers[action_index].writerow(["Epoch", "Training MSE", "Validation MSE"])

            # create evaluation file writer
            submodel_evaluation_path = os.path.join(self.__episode_path, str(action_index) + '_evaluation.csv')
            evaluation_file = open(submodel_evaluation_path, 'w', newline='')
            self.__model_evaluate_writers[action_index] = csv.writer(evaluation_file)
            self.__model_evaluate_writers[action_index].writerow(["Step", "Evaluation MSE"])

        self._episode = value

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, value):
        self._step = value

    def log_action_data(self, action_record):
        self.__reward_writer.writerow([self.step, action_record[3]])

    def print(self, string, log_level = 0):
        if self.__logging_enabled and log_level.value >= Settings.LOG_LEVEL:
            print(string)

    def log_training(self, action_index, training_history):
        # write training history
        for epoch in range(len(training_history.epoch)):
            epoch_mse = training_history.history["mse"][epoch]
            epoch_val_mse = training_history.history["val_mse"][epoch]
            # TOOD: add offset for epoch
            self.__model_train_writers[action_index].writerow([epoch, epoch_mse, epoch_val_mse])

    def log_evaluation(self, action_index, mse):
        self.__model_evaluate_writers[action_index].writerow([self.episode, mse])

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
