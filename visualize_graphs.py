import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.python.summary.summary_iterator import summary_iterator

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

HYPOTHETICAL_OPTIMAL_REWARD = 11.976

def plot_for_scenario(path, scenario_save, scenario_title, length_max = 300):
    steps = []
    rewards = []
    lengths = []
    # This example supposes that the events file contains summaries with a
    # summary value tag 'loss'.  These could have been added by calling
    # `add_summary()`, passing the output of a scalar summary op created with
    # with: `tf.scalar_summary(['loss'], loss_tensor)`.
    for e in summary_iterator(path):
        for v in e.summary.value:
            # if v.tag == 'rollout/ep_rew_mean' or v.tag == 'rollout/ep_len_mean':
            if v.tag == 'rollout/ep_rew_mean':
                # print(f"{v=}")
                steps.append(e.step)
                rewards.append(v.simple_value)
            elif v.tag == 'rollout/ep_len_mean':
                lengths.append(v.simple_value)
                # print(f"{e.step=}")
                # print(f"{v.simple_value=}")


    figsize =  (6,4)
    #-----------------------------------------REWARD PLOT------------------------------------
    hypothetical_optimal_human = np.full(len(steps), length_max*HYPOTHETICAL_OPTIMAL_REWARD)

    plt.figure(figsize=figsize, dpi=300, layout='constrained')
    plt.plot(steps, rewards, color='lightskyblue', label='PPO LSTM Raw Values', alpha=0.5)

    rewards_smoothed = pd.DataFrame(rewards)
    rewards_smoothed = rewards_smoothed.rolling(window=500).mean()

    plt.plot(steps, rewards_smoothed, color='royalblue', label='PPO LSTM Running Average')

    plt.plot(steps, hypothetical_optimal_human, color='black', label='Hypothetical Optimal Human', alpha=0.7)

    

    #Use scientific magnitudes
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    # plt.gca().ticklabel_format(useMathText=True)

    plt.title(f"Mean Episode Reward for the {scenario_title} Scenario")
    plt.xlabel("Step")
    plt.ylabel("Episode Reward")
    plt.legend()
    # plt.show()
    plt.savefig(f"./figures_journal/{scenario_save}_episode_reward.png")


    #-----------------------------------------LENGTH PLOT------------------------------------
    length_max_line = np.full(len(steps), length_max)


    plt.figure(figsize=figsize, dpi=300, layout='constrained')
    plt.plot(steps, lengths, color='lightskyblue', label='PPO LSTM Raw Values', alpha=0.5)

    lengths_smoothed = pd.DataFrame(lengths)
    lengths_smoothed = lengths_smoothed.rolling(window=500).mean()

    plt.plot(steps, lengths_smoothed, color='royalblue', label='PPO LSTM Running Average')

    plt.plot(steps, length_max_line, color='black', label='Maximum Episode Length', alpha=0.7)


    index_max_length = max(range(len(lengths)), key=lengths.__getitem__)
    index_max_reward = max(range(len(rewards)), key=rewards.__getitem__)
    print(f"{scenario_title}. {max(lengths)=} occured at {steps[index_max_length]}, with mean reward {rewards[index_max_length]}")
    print(f"{scenario_title}. {max(rewards)=} occured at {steps[index_max_reward]}, with mean length {lengths[index_max_reward]}")

    #Use scientific magnitudes
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    # plt.gca().ticklabel_format(useMathText=True)

    plt.title(f"Mean Episode Length for the {scenario_title} Scenario")
    plt.xlabel("Step")
    plt.ylabel("Episode Length")
    plt.legend()
    # plt.show()
    plt.savefig(f"./figures_journal/{scenario_save}_episode_length.png")



path_to_events_file_basic_tracking = "E:\\Repos\\PTZ_drone_tracking\\tensorboard_logging\\Tracking_Basic_lr_0.0001\\RecurrentPPO_2\\events.out.tfevents.1704236115.DESKTOP-9UBP6R9.29444.0"
plot_for_scenario(path_to_events_file_basic_tracking, "basic_tracking", "Basic Tracking")

path_to_events_file_dynamic_tracking_a = "E:\\Repos\\PTZ_drone_tracking\\tensorboard_logging\\Tracking_Dynamic_lr_0.0002_retrain\\RecurrentPPO_1\\events.out.tfevents.1704404262.DESKTOP-9UBP6R9.20228.0"
plot_for_scenario(path_to_events_file_dynamic_tracking_a, "dynamic_tracking_a", "Dynamic Tracking A", length_max=301)

# path_to_events_file_dynamic_tracking_b = "E:\\Repos\\PTZ_drone_tracking\\tensorboard_logging\\Tracking_Dynamic_lr_0.0001_lower_speed\\RecurrentPPO_1\\events.out.tfevents.1704475851.DESKTOP-9UBP6R9.34896.0"
path_to_events_file_dynamic_tracking_b = "E:\\Repos\\PTZ_drone_tracking\\tensorboard_logging\\Tracking_Dynamic_lr_7.5e-05_loaded_model\\RecurrentPPO_2\\events.out.tfevents.1705352803.DESKTOP-9UBP6R9.34380.0"
plot_for_scenario(path_to_events_file_dynamic_tracking_b, "dynamic_tracking_b", "Dynamic Tracking B")

path_to_events_file_obstacle_tracking = "E:\\Repos\\PTZ_drone_tracking\\tensorboard_logging\\Tracking_Obstacle_lr_7.5e-05_loaded_model\\RecurrentPPO_2\\events.out.tfevents.1705233613.DESKTOP-9UBP6R9.32672.0"
plot_for_scenario(path_to_events_file_obstacle_tracking, "obstacle_tracking", "Obstacle Tracking", length_max=450)


# #-----------------------------------------REWARD PLOT------------------------------------
# plt.figure(figsize=figsize, dpi=100, layout='constrained')
# plt.plot(steps, rewards, color='lightskyblue', label='PPO LSTM Raw Values', alpha=0.5)

# rewards_smoothed = pd.DataFrame(rewards)
# rewards_smoothed = rewards_smoothed.rolling(window=500).mean()

# plt.plot(steps, rewards_smoothed, color='royalblue', label='PPO LSTM Running Average')

# #Use scientific magnitudes
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# # plt.gca().ticklabel_format(useMathText=True)

# plt.title("Mean Episode Reward for the Simple Tracking Scenario")
# plt.xlabel("Step")
# plt.ylabel("Episode Reward")
# plt.legend()
# # plt.show()
# plt.savefig("./figures_journal/basic_tracking_episode_reward.png")


# #-----------------------------------------LENGTH PLOT------------------------------------
# LENGTH_MAX = np.full(len(steps), 300)


# plt.figure(figsize=figsize, dpi=100, layout='constrained')
# plt.plot(steps, lengths, color='lightskyblue', label='PPO LSTM Raw Values', alpha=0.5)

# lengths_smoothed = pd.DataFrame(lengths)
# lengths_smoothed = lengths_smoothed.rolling(window=500).mean()

# plt.plot(steps, lengths_smoothed, color='royalblue', label='PPO LSTM Running Average')

# plt.plot(steps, LENGTH_MAX, color='black', label='Maximum Possible Value', alpha=0.7)

# #Use scientific magnitudes
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# # plt.gca().ticklabel_format(useMathText=True)

# plt.title("Mean Episode Length for the Simple Tracking Scenario")
# plt.xlabel("Step")
# plt.ylabel("Episode Length")
# plt.legend()
# # plt.show()
# plt.savefig("./figures_journal/basic_tracking_episode_length.png")