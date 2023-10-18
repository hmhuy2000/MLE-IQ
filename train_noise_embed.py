import datetime
import os
import random
import time
from collections import deque
from itertools import count
import types

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter

from make_envs import make_env
from embed import embed
from agent import make_agent
from dataset.memory import Memory
from utils.utils import eval_mode,evaluate,soft_update,get_concat_samples

def get_args(cfg: DictConfig):
    cfg.device = "cuda"
    cfg.hydra_base_dir = os.getcwd()
    return cfg

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)
    
    wandb.init(project='offline-mujoco', settings=wandb.Settings(_disable_stats=True), \
        group=args.env.name,job_type=f'MLE-{args.expert.demos}', name=f'{args.seed}', entity='hmhuy')
    print(OmegaConf.to_yaml(args))
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == 'cuda' and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    eval_env = make_env(args)
    eval_env.seed(args.seed + 10)
    REPLAY_MEMORY = int(args.env.replay_mem)
    INITIAL_MEMORY = int(args.env.initial_mem)
    EPISODE_STEPS = int(args.env.eps_steps)
    EPISODE_WINDOW = int(args.env.eps_window)
    LEARN_STEPS = int(args.env.learn_steps)
    INITIAL_STATES = 128
    obs_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]
    agent = make_agent(eval_env, args)
    agent.embed = embed(obs_dim=obs_dim,action_dim=action_dim,args=args).to(device)
    agent.embed.load(f'/home/huy/codes/2023/MLE-IQ/pretrained/ant_expert-v2-{args.embed.latent_dim}')
    agent.embed.eval()
    # Load expert data
    expert_memory_replay = Memory(REPLAY_MEMORY//2, args.seed)
    expert_memory_replay.load(hydra.utils.to_absolute_path(f'experts/{args.env.demo}'),
                              num_trajs=args.expert.demos,
                              sample_freq=args.expert.subsample_freq,
                              seed=args.seed + 42)
    print(f'--> Expert memory size: {expert_memory_replay.size()}')
    sub_optimal_memory_replay = Memory(REPLAY_MEMORY//2, args.seed+1)
    sub_optimal_memory_replay.load(hydra.utils.to_absolute_path(f'experts/{args.env.sub_optimal_demo}'),
                              num_trajs=args.env.num_sub_optimal_demo,
                              sample_freq=args.expert.subsample_freq,
                              seed=args.seed + 42)
    print(f'--> sub optimal Expert memory size: {sub_optimal_memory_replay.size()}')

    online_memory_replay = Memory(REPLAY_MEMORY//2, args.seed+1)

    learn_steps = 0
    
    def compute_energy(embed_1, embed_2):
        squared_diff = torch.square(embed_1.unsqueeze(1) - embed_2.unsqueeze(0))
        energies = torch.sum(-squared_diff, dim=-1)
        return energies
    
    # env = make_env(args)
    # env.seed(args.seed)
    # state = env.reset()
    # for _ in range(10):
    #     with torch.no_grad():
    #         state, next_state, action, _,_ = sub_optimal_memory_replay.get_samples(1000, device)
    #         latent = agent.embed.state_function(torch.tensor(state,dtype=torch.float).to(device))
    #         next_latent = agent.embed.latent_action_function(torch.cat((latent,action),dim=-1))
    #         actual_latent = agent.embed.state_function(torch.tensor(next_state,dtype=torch.float).to(device))
    #         print(next_latent.shape,actual_latent.shape)
    #         energy = compute_energy(next_latent,actual_latent)
    #         pos = torch.diagonal(energy, dim1=-2, dim2=-1)
    #         neg = torch.logsumexp(energy, dim=-1)
    #         print(pos.mean(),neg.mean())
    #         raise
    
    traj_len = deque(maxlen = 100)
    length = 0
    done = True
    for iter in count():
        for _ in range(10):
            if (done):
                traj_len.append(length)
                length = 0
                sub_state= sub_optimal_memory_replay.get_random_initial(device=device)
                expert_state = sub_optimal_memory_replay.get_random_initial(device=device)
                if (np.random.rand()<0.5):
                    latent = agent.embed.state_function(sub_state)
                else:
                    latent = agent.embed.state_function(expert_state)
                done = False
            with eval_mode(agent):
                action = agent.actor(latent).mean
                next_latent = agent.embed.latent_action_function(torch.cat((latent,action),dim=-1))
            length += 1
            done = agent.embed.done_function(torch.cat((latent,next_latent),dim=-1))
            done = (done[:, 1] < done[:, 0]).float()[0].item()
            
        
            online_memory_replay.add((latent.detach().cpu().numpy()[0],
                                    next_latent.detach().cpu().numpy()[0],
                                    action.detach().cpu().numpy()[0], 0, done))
            latent = next_latent
        if online_memory_replay.size() <= INITIAL_MEMORY:
            continue
        
        info = {}
        agent.iq_update = types.MethodType(iq_update, agent)
        agent.iq_update_critic = types.MethodType(iq_update_critic, agent)
        losses = agent.iq_update(online_memory_replay,
                                    expert_memory_replay, learn_steps)
        if (iter%5000 == 0):
            info.update(losses)
            eval_returns, eval_timesteps = evaluate(agent, eval_env, num_episodes=args.eval.eps)
            returns = np.mean(eval_returns)
            num_steps = np.mean(eval_timesteps)
            info['Eval/return'] = returns
            info['Eval/num_steps'] = num_steps
            info['Train/num_steps'] = np.mean(traj_len)
            learn_steps += 1  # To prevent repeated eval at timestep 0
            print('EVAL\titer {}\tAverage reward: {:.2f}\t'.format(iter, returns))
            try:
                wandb.log(info,step = learn_steps)
            except:
                print(info)
   
def iq_update_critic(self, policy_batch, expert_batch):
    args = self.args
    batch = get_concat_samples(policy_batch, expert_batch, args)
    obs, next_obs, action = batch[0:3]

    agent = self
    current_V = self.getV(obs)
    if args.train.use_target:
        with torch.no_grad():
            next_V = self.get_targetV(next_obs)
    else:
        next_V = self.getV(next_obs)

    current_Q = self.critic(obs, action)
    critic_loss, loss_dict = iq_loss(agent, current_Q, current_V, next_V, batch)

    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()
    return loss_dict

def iq_update(self, online_buffer, expert_buffer, step):
    online_batch = online_buffer.get_samples(self.batch_size, self.device)
    expert_batch = expert_buffer.get_samples(self.batch_size, self.device)
    
    expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch
    expert_obs = self.embed.state_function(expert_obs).detach()
    expert_next_obs = self.embed.state_function(expert_next_obs).detach()
    expert_batch = expert_obs, expert_next_obs, expert_action, expert_reward, expert_done

    losses = self.iq_update_critic(online_batch, expert_batch)
    if self.actor and step % self.actor_update_frequency == 0:
        if self.args.offline:
            if (self.first_log):
                print('actor training: only expert')
            obs = expert_batch[0]
        else:
            if (self.first_log):
                print('actor training: with sub optimal')
            obs = torch.cat([online_batch[0], expert_batch[0]], dim=0)

        if self.args.num_actor_updates:
            for i in range(self.args.num_actor_updates):
                actor_alpha_losses = self.update_actor_and_alpha(obs)
        losses.update(actor_alpha_losses)

    if step % self.critic_target_update_frequency == 0:
        if self.args.train.soft_update:
            soft_update(self.critic_net, self.critic_target_net,
                        self.critic_tau)
        else:
            raise
    self.first_log=False
    
    return losses

def iq_loss(agent, current_Q, current_v, next_v, batch):
    args = agent.args
    gamma = agent.gamma
    obs, next_obs, action, env_reward, done, is_expert = batch

    loss_dict = {}
    loss_dict.update({
        'info/expert_Q':current_Q[is_expert].mean().item(),
        'info/add_Q':current_Q[~is_expert].mean().item(),
        'info/expert_v':current_v[is_expert].mean().item(),
        'info/add_v':current_v[~is_expert].mean().item(),
    })
    y = (1 - done) * gamma * next_v
    reward = (current_Q - y)[is_expert]
    with torch.no_grad():
        log_pi = agent.actor.get_log_prob(obs=obs,action=action)
        loss_dict.update({
            'info/expert_reward':(current_Q - y)[is_expert].mean().item(),
            'info/add_reward':(current_Q - y)[~is_expert].mean().item(),
            'info/expert_log_prob':log_pi[is_expert].mean().item(),
            'info/add_log_prob':log_pi[~is_expert].mean().item(),
        })
    loss = -reward.mean()
    loss_dict['softq_loss'] = loss.item()

    if args.method.loss == "value":
        if (agent.first_log):
            print('value loss: value (with sub optimal)')
        value_loss = (current_v - y).mean()
        loss += value_loss
        loss_dict['value_loss'] = value_loss.item()
    else:
        raise ValueError(f'This sampling method is not implemented: {args.method.type}')

    if args.method.regularize:
        if (agent.first_log):
            print('regularize loss: chi2 (with sub optimal)')
        # Use χ2 divergence (calculate the regularization term for IQ loss using expert and policy states) (works online)
        y = (1 - done) * gamma * next_v

        reward = current_Q - y
        chi2_loss = 1/(4 * args.method.alpha) * (reward**2).mean()
        loss += chi2_loss
        loss_dict['regularize_loss'] = chi2_loss.item()

    loss_dict['total_loss'] = loss.item()
    return loss, loss_dict

def save(agent, epoch, args, output_dir='results'):
    if epoch % args.save_interval == 0:
        if args.method.type == "sqil":
            name = f'sqil_{args.env.name}'
        else:
            name = f'iq_{args.env.name}'

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        agent.save(f'{output_dir}/{args.agent.name}_{name}')

if __name__ == '__main__':
    main()