import numpy as np
from scipy.spatial import distance
import pandas as pd


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


def prepare_data():
    topo_data = pd.read_csv('data/topo.csv', header=None)

    data = topo_data[6].to_numpy()
    import random
    target = random.sample(range(data.shape[0]), k=150)

    topo_string = ','.join([x for x in data])
    target_string = ', '.join([str(x) for x in target])

    old_dat = pd.read_csv('data/formal_data.csv', header=0)
    row = old_dat.iloc[0].copy()
    new_data = pd.DataFrame(columns=old_dat.columns)
    # new_data.append({
    #     'No. Data': 1,
    #     'target': target_string,
    #     'node_pos': topo_string,
    #     'energy': old_dat.energy,
    #     'commRange': 15,
    #     'freq': old_dat.freq,
    #     'charge_pos': old_dat.charge_pos,
    #     'chargeRange': old_dat.chargeRange,
    #     'velocity': old_dat.velocity,
    #     'base': old_dat.base,
    #     'depot': old_dat.velocity,
    # })
    row['commRange'] = 15
    row['target'] = target_string
    row['node_pos'] = topo_string
    new_data = new_data.append(row.to_dict(), ignore_index=True)

    new_data.to_csv('data/formal_data_3.csv')


if __name__ == '__main__':
    prepare_data()
