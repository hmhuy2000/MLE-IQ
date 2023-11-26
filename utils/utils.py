import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def evaluate(actor, env,shift,scale,env_name, num_episodes=100, vis=True):
    """Evaluates the policy.
    Args:
      actor: A policy to evaluate.
      env: Environment to evaluate the policy on.
      num_episodes: A number of episodes to average the policy on.
    Returns:
      Averaged reward and a total number of steps.
    """
    total_timesteps = []
    total_returns = []

    while len(total_returns) < num_episodes:
        state = env.reset()
        done = False

        with eval_mode(actor):
            while not done:
                if ('Ant' in env_name):
                    state = state[:27]
                state = (state + shift) * scale
                action = actor.choose_action(state, sample=False)
                next_state, reward, done, info = env.step(action)
                state = next_state

                if 'episode' in info.keys():
                    total_returns.append(info['episode']['r'])
                    total_timesteps.append(info['episode']['l'])

    return total_returns, total_timesteps


def weighted_softmax(x, weights):
    x = x - torch.max(x, dim=0)[0]
    return weights * torch.exp(x) / torch.sum(
        weights * torch.exp(x), dim=0, keepdim=True)


def soft_update(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def hard_update(source, target):
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(param.data)


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def get_concat_samples(policy_batch, expert_batch, args):
    online_batch_state, online_batch_next_state, online_batch_action, online_batch_reward, online_batch_done = policy_batch

    expert_batch_state, expert_batch_next_state, expert_batch_action, expert_batch_reward, expert_batch_done = expert_batch

    if args.method.type == "sqil":
        # convert policy reward to 0
        online_batch_reward = torch.zeros_like(online_batch_reward)
        # convert expert reward to 1
        expert_batch_reward = torch.ones_like(expert_batch_reward)

    batch_state = torch.cat([online_batch_state, expert_batch_state], dim=0)
    batch_next_state = torch.cat(
        [online_batch_next_state, expert_batch_next_state], dim=0)
    batch_action = torch.cat([online_batch_action, expert_batch_action], dim=0)
    batch_reward = torch.cat([online_batch_reward, expert_batch_reward], dim=0)
    batch_done = torch.cat([online_batch_done, expert_batch_done], dim=0)
    is_expert = torch.cat([torch.zeros_like(online_batch_reward, dtype=torch.bool),
                           torch.ones_like(expert_batch_reward, dtype=torch.bool)], dim=0)

    return batch_state, batch_next_state, batch_action, batch_reward, batch_done, is_expert

def get_concat_with_add_samples(policy_batch, add_batches, args):
    online_batch_state, online_batch_next_state, online_batch_action, online_batch_reward, online_batch_done = policy_batch
    
    batch_state = [online_batch_state]
    batch_next_state = [online_batch_next_state]
    batch_action = [online_batch_action]
    batch_reward  =[online_batch_reward]
    batch_done = [online_batch_done]
    for batch in add_batches:
        add_batch_state, add_batch_next_state, add_batch_action, add_batch_reward, add_batch_done = batch
        assert online_batch_reward.shape ==add_batch_reward.shape,NotImplementedError
        batch_state.append(add_batch_state)
        batch_next_state.append(add_batch_next_state)
        batch_action.append(add_batch_action)
        batch_reward.append(add_batch_reward)
        batch_done.append(add_batch_done)
    batch_state = torch.cat(batch_state, dim=0)
    batch_next_state = torch.cat(batch_next_state, dim=0)
    batch_action = torch.cat(batch_action, dim=0)
    batch_reward = torch.cat(batch_reward, dim=0)
    batch_done = torch.cat(batch_done, dim=0)
    is_pi = [torch.ones_like(online_batch_reward, dtype=torch.bool)]
    for _ in range(len(add_batches)):
        is_pi.append(torch.zeros_like(online_batch_reward, dtype=torch.bool))
    is_pi = torch.cat(is_pi, dim=0)
    is_adds = []
    for id in range(len(add_batches)):
        is_add = [torch.zeros_like(online_batch_reward, dtype=torch.bool)]
        for idx in range(len(add_batches)):
            if (idx == id):
                is_add.append(torch.ones_like(online_batch_reward, dtype=torch.bool))
            else:
                is_add.append(torch.zeros_like(online_batch_reward, dtype=torch.bool))
        is_add = torch.cat(is_add,dim=0)
        is_adds.append(is_add)
    return batch_state, batch_next_state, batch_action, batch_reward, batch_done,\
        is_pi, is_adds

def save_state(tensor, path, num_states=5):
    """Show stack framed of images consisting the state"""

    tensor = tensor[:num_states]
    B, C, H, W = tensor.shape
    images = tensor.reshape(-1, 1, H, W).cpu()
    save_image(images, path, nrow=num_states)
    # make_grid(images)

def concat_data(add_batches,reward_arr, args):
    batch_state = []
    batch_next_state = []
    batch_action = []
    batch_reward  =[]
    batch_done = []
    
    for reward,batch in zip(reward_arr,add_batches):
        add_batch_state, add_batch_next_state, add_batch_action, add_batch_reward, add_batch_done = batch
        batch_state.append(add_batch_state)
        batch_next_state.append(add_batch_next_state)
        batch_action.append(add_batch_action)
        batch_reward.append(torch.full_like(add_batch_reward,reward))
        batch_done.append(add_batch_done)
    batch_state = torch.cat(batch_state, dim=0)
    batch_next_state = torch.cat(batch_next_state, dim=0)
    batch_action = torch.cat(batch_action, dim=0)
    batch_reward = torch.cat(batch_reward, dim=0)
    batch_done = torch.cat(batch_done, dim=0)
    return batch_state, batch_next_state, batch_action, batch_reward, batch_done


def average_dicts(dict1, dict2):
    return {key: 1/2 * (dict1.get(key, 0) + dict2.get(key, 0))
                     for key in set(dict1) | set(dict2)}