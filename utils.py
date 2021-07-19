from math import dist
import numpy as np
from scipy.spatial import distance


def updateMemories(q_learning, deep_qlearning):
    next_action = q_learning.state
    reward = q_learning.reward_dqn
    input_state_dqn = np.reshape(q_learning.input_state_dqn, [
                                 1, deep_qlearning.state_size])
    print("reward: {}".format(reward))

    # update temporary memories
    deep_qlearning.memorize(input_state_dqn, next_action,
                            reward, input_state_dqn)


def updateNextAction(deep_qlearning, next_state):
    if len(deep_qlearning.memory) > 0:
        last_memory = list(deep_qlearning.memory[-1])
        last_memory[3] = next_state
        deep_qlearning.memory[-1] = tuple(last_memory)

        # state, action, reward, next_state
        last_piority = deep_qlearning.priority[-1]
        last_piority = deep_qlearning.prioritize(
            last_memory[0], next_state, last_memory[1], last_memory[2], alpha=0.6)
        deep_qlearning.priority[-1] = last_piority


def _build_input_state(network, t):
    list_state = []
    # normalize data to pass network
    energies = [nd.energy for nd in network.node]
    dists = [distance.euclidean(nd.location, network.mc.current) for nd in network.node]
    start_time_windows = [nd.window_time[0] for nd in network.node]
    end_time_windows = [nd.window_time[1] for nd in network.node]
    tw_opens = [0 if nd.window_time[0] > t else 1 for nd in network.node]
    # max_energy = max(energies)
    # min_energy = min(energies)
    # max_avg = max(avg_energies)
    # min_avg = min(avg_energies)

    # list_energy_normalize = [(nd.energy / max_energy )  for nd in network.node]
    # list_avg_energy_normalize = [(nd.avg_energy / max_avg) for nd in network.node]

    list_state.append(network.mc.energy)
    list_state.extend(energies)
    list_state.extend(dists)
    list_state.extend(start_time_windows)
    list_state.extend(end_time_windows)
    list_state.extend(tw_opens)

    return np.array(list_state)
