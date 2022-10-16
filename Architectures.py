from enum import Enum
import tensorflow as tf

import Settings

class Architectures(Enum):
  FC_1_16 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_dim=Settings.OBSERVATION_SIZE),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(Settings.OBSERVATION_SIZE, activation='sigmoid')
  ])

  FC_1_32 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_dim=Settings.OBSERVATION_SIZE),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(Settings.OBSERVATION_SIZE, activation='sigmoid')
  ])

  FC_3_32 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_dim=Settings.OBSERVATION_SIZE),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(Settings.OBSERVATION_SIZE, activation='sigmoid')
  ])

  FC_3_64 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_dim=Settings.OBSERVATION_SIZE),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(Settings.OBSERVATION_SIZE, activation='sigmoid')
  ])

  FC_5_32 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_dim=Settings.OBSERVATION_SIZE),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(Settings.OBSERVATION_SIZE, activation='sigmoid')
  ])

  FC_5_64 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_dim=Settings.OBSERVATION_SIZE),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(Settings.OBSERVATION_SIZE, activation='sigmoid')
  ])

  REWARD_1_32 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_dim=Settings.OBSERVATION_SIZE),
    tf.keras.layers.Dense(1)
  ])

  REWARD_1_64 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_dim=Settings.OBSERVATION_SIZE),
    tf.keras.layers.Dense(1)
  ])

  REWARD_1_128 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_dim=Settings.OBSERVATION_SIZE),
    tf.keras.layers.Dense(1)
  ])

  REWARD_1_256 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_dim=Settings.OBSERVATION_SIZE),
    tf.keras.layers.Dense(1)
  ])

  REWARD_2_16 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_dim=Settings.OBSERVATION_SIZE),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  REWARD_2_32 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_dim=Settings.OBSERVATION_SIZE),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  REWARD_2_64 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_dim=Settings.OBSERVATION_SIZE),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  REWARD_3_32 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_dim=Settings.OBSERVATION_SIZE),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  REWARD_3_64 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_dim=Settings.OBSERVATION_SIZE),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  REWARD_3_128 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_dim=Settings.OBSERVATION_SIZE),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
  ])
