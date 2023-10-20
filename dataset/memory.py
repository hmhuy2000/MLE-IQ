from collections import deque
import numpy as np
import random
import torch
from tqdm import trange
from dataset.expert_dataset import ExpertDataset


class Memory(object):
    def __init__(self, memory_size: int, seed: int = 0) -> None:
        random.seed(seed)
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()

    def save(self, path):
        b = np.asarray(self.buffer)
        print(b.shape)
        np.save(path, b)

    def get_random_traj(self,device):
        rand = np.random.randint(0, len(self.traj_states))
        states = torch.as_tensor(self.traj_states[rand], dtype=torch.float, device=device)
        actions = torch.as_tensor(self.traj_actions[rand], dtype=torch.float, device=device)
        next_states = torch.as_tensor(self.traj_next_states[rand], dtype=torch.float, device=device)
        return states,actions,next_states

    def load(self, path, num_trajs, sample_freq, seed,
             save_trajs=False):
        # If path has no extension add npy
        if not (path.endswith("pkl") or path.endswith("hdf5")):
            path += '.npy'
        data = ExpertDataset(path, num_trajs, sample_freq, seed)
        self.initial = data.get_initial()
        self.memory_size = data.__len__()
        self.buffer = deque(maxlen=self.memory_size)
        if (save_trajs):
            self.traj_states = data.trajectories['states']
            self.traj_actions = data.trajectories['actions']
            self.traj_next_states = data.trajectories['next_states']
        for i in range(len(data)):
            self.add(data[i])

    def get_random_initial(self,device):
        rand = np.random.randint(0, len(self.initial))
        return torch.as_tensor(self.initial[rand], dtype=torch.float, device=device).unsqueeze(0)

    def get_samples(self, batch_size, device):
        batch = self.sample(batch_size, False)

        batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(
            *batch)

        batch_state = np.array(batch_state)
        batch_next_state = np.array(batch_next_state)
        batch_action = np.array(batch_action)

        batch_state = torch.as_tensor(batch_state, dtype=torch.float, device=device)
        batch_next_state = torch.as_tensor(batch_next_state, dtype=torch.float, device=device)
        batch_action = torch.as_tensor(batch_action, dtype=torch.float, device=device)
        if batch_action.ndim == 1:
            batch_action = batch_action.unsqueeze(1)
        batch_reward = torch.as_tensor(batch_reward, dtype=torch.float, device=device).unsqueeze(1)
        batch_done = torch.as_tensor(batch_done, dtype=torch.float, device=device).unsqueeze(1)

        return batch_state, batch_next_state, batch_action, batch_reward, batch_done

    def sample_x_trans(self, batch_size,history):
        indexes = np.random.choice(np.arange(len(self.buffer)-history + 1), size=batch_size, replace=False)
        return [[self.buffer[i+j] for i in indexes]
                                for j in range(history)]

    def get_samples_x_trans(self, batch_size,history, device):
        batch_ls = self.sample_x_trans(batch_size,history)
        
        action_ls = []
        for idx in range(len(batch_ls)):
            batch = batch_ls[idx]
            batch_state, batch_next_state, batch_action, _, _ = zip(*batch)
            if (idx == 0):
                initial_states = np.array(batch_state)
                initial_states = torch.as_tensor(initial_states, dtype=torch.float, device=device)
            if (idx == len(batch_ls)-1):
                final_states = np.array(batch_next_state)
                final_states = torch.as_tensor(final_states, dtype=torch.float, device=device)
            batch_action = np.array(batch_action)
            batch_action = torch.as_tensor(batch_action, dtype=torch.float, device=device)
            action_ls.append(batch_action)

        return initial_states,final_states,action_ls
