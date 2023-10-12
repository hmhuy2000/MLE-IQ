from typing import Any, Dict, IO, List, Tuple

import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import os


class ExpertDataset(Dataset):
    """Dataset for expert trajectories.

    Assumes expert dataset is a dict with keys {states, actions, rewards, lengths} with values containing a list of
    expert attributes of given shapes below. Each trajectory can be of different length.

    Expert rewards are not required but can be useful for evaluation.

        shapes:
            expert["states"]  =  [num_experts, traj_length, state_space]
            expert["actions"] =  [num_experts, traj_length, action_space]
            expert["rewards"] =  [num_experts, traj_length]
            expert["lengths"] =  [num_experts]
    """

    def __init__(self,
                 expert_location: str,
                 num_trajectories: int = 4,
                 subsample_frequency: int = 20,
                 seed: int = 0):
        """Subsamples an expert dataset from saved expert trajectories.

        Args:
            expert_location:          Location of saved expert trajectories.
            num_trajectories:         Number of expert trajectories to sample (randomized).
            subsample_frequency:      Subsamples each trajectory at specified frequency of steps.
            deterministic:            If true, sample determinstic expert trajectories.
        """
        all_trajectories = load_trajectories(expert_location, num_trajectories, seed)
        self.trajectories = {}

        # Randomize start index of each trajectory for subsampling
        # start_idx = torch.randint(0, subsample_frequency, size=(num_trajectories,)).long()

        # Subsample expert trajectories with every `subsample_frequency` step.
        for k, v in all_trajectories.items():
            data = v

            if k != "lengths":
                samples = []
                for i in range(num_trajectories):
                    samples.append(data[i][0::subsample_frequency])
                self.trajectories[k] = samples
            else:
                # Adjust the length of trajectory after subsampling
                self.trajectories[k] = np.array(data) // subsample_frequency

        self.i2traj_idx = {}
        self.length = self.trajectories["lengths"].sum().item()

        del all_trajectories  # Not needed anymore
        traj_idx = 0
        i = 0

        # Convert flattened index i to trajectory indx and offset within trajectory
        self.get_idx = []

        for _j in range(self.length):
            while self.trajectories["lengths"][traj_idx].item() <= i:
                i -= self.trajectories["lengths"][traj_idx].item()
                traj_idx += 1

            self.get_idx.append((traj_idx, i))
            i += 1

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.length

    def __getitem__(self, i):
        traj_idx, i = self.get_idx[i]
        return (self.trajectories["states"][traj_idx][i],
                self.trajectories["next_states"][traj_idx][i],
                self.trajectories["actions"][traj_idx][i],
                self.trajectories["rewards"][traj_idx][i],
                self.trajectories["dones"][traj_idx][i])


def load_trajectories(expert_location: str,
                      num_trajectories: int = 10,
                      seed: int = 0) -> Dict[str, Any]:
    """Load expert trajectories

    Args:
        expert_location:          Location of saved expert trajectories.
        num_trajectories:         Number of expert trajectories to sample (randomized).
        deterministic:            If true, random behavior is switched off.

    Returns:
        Dict containing keys {"states", "lengths"} and optionally {"actions", "rewards"} with values
        containing corresponding expert data attributes.
    """
    if os.path.isfile(expert_location):
        if (expert_location.endswith("hdf5")):
            with open(expert_location, 'rb') as f:
                hdf_trajs = read_file(expert_location, f)
            starts_timeout = np.where(np.array(hdf_trajs['timeouts'])>0)[0].tolist()
            starts_done = np.where(np.array(hdf_trajs['terminals'])>0)[0].tolist()
            starts = [-1]+starts_timeout+starts_done
            starts = list(dict.fromkeys(starts))
            starts.sort()
            
            rng = np.random.RandomState(seed)
            perm = np.arange(len(starts)-1)
            perm = rng.permutation(perm)
            idx = perm[:num_trajectories]
            trajs = {}
            trajs['lengths'] = [starts[idx[i]+1] - starts[idx[i]] for i in range(len(idx))]
            trajs['dones'] = [hdf_trajs['terminals'][starts[idx[i]]+1:starts[idx[i]+1]+1]
                              for i in range(len(idx))]
            trajs['rewards'] = [hdf_trajs['rewards'][starts[idx[i]]+1:starts[idx[i]+1]+1]
                              for i in range(len(idx))]
            trajs['states'] = [hdf_trajs['observations'][starts[idx[i]]+1:starts[idx[i]+1]+1]
                              for i in range(len(idx))]
            trajs['next_states'] = [hdf_trajs['next_observations'][starts[idx[i]]+1:starts[idx[i]+1]+1]
                              for i in range(len(idx))]
            trajs['actions'] = [hdf_trajs['actions'][starts[idx[i]]+1:starts[idx[i]+1]+1]
                              for i in range(len(idx))]
            reward_arr = [np.sum(trajs['rewards'][i]) for i in range(len(trajs['rewards']))]
            print(f'expert: {expert_location}, {len(perm)} trajectories')
            print(f'return: {np.mean(reward_arr)}, max:{np.max(reward_arr)}, min:{np.min(reward_arr)}')
            print(f'mean episode length: {np.mean(trajs["lengths"])}')
            return trajs  
    
        else:
            with open(expert_location, 'rb') as f:
                trajs = read_file(expert_location, f)
            rng = np.random.RandomState(seed)
            # Sample random `num_trajectories` experts.
            perm = np.arange(len(trajs["states"]))
            perm = rng.permutation(perm)

            idx = perm[:num_trajectories]
            for k, v in trajs.items():
                trajs[k] = [v[i] for i in idx]

    else:
        raise ValueError(f"{expert_location} is not a valid path")
    return trajs


def read_file(path: str, file_handle: IO[Any]) -> Dict[str, Any]:
    """Read file from the input path. Assumes the file stores dictionary data.

    Args:
        path:               Local or S3 file path.
        file_handle:        File handle for file.

    Returns:
        The dictionary representation of the file.
    """
    if path.endswith("pt"):
        data = torch.load(file_handle)
    elif path.endswith("pkl"):
        data = pickle.load(file_handle)
    elif path.endswith("hdf5"):
        import h5py
        data = h5py.File(path, 'r')
    elif path.endswith("npy"):
        data = np.load(file_handle, allow_pickle=True)
        if data.ndim == 0:
            data = data.item()
    else:
        raise NotImplementedError
    return data
