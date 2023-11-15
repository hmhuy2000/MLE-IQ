import datetime
import os
import random
import time
from collections import deque
from itertools import count
import types
from tqdm import trange

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import nn
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter
from torch.optim import Adam
from make_envs import make_env
from agent import make_agent
from dataset.memory import Memory
from utils.utils import eval_mode,evaluate,soft_update,concat_data

def get_args(cfg: DictConfig):
    cfg.device = "cuda"
    cfg.hydra_base_dir = os.getcwd()
    return cfg

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)
    
    run_name = f'Ours'
    for expert_dir,num_expert in zip(args.env.sub_optimal_demo,args.env.num_sub_optimal_demo):
        run_name += f'-{expert_dir.split(".")[0].split("/")[-1]}({num_expert})'
    wandb.init(project=f'offline-{args.env.name}', settings=wandb.Settings(_disable_stats=True), \
        group='offline',
        job_type=run_name,
        name=f'{args.seed}', entity='hmhuy')
    print(OmegaConf.to_yaml(args))
    print(run_name)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == 'cuda' and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    env = make_env(args)
    eval_env = make_env(args)
    env.seed(args.seed)
    eval_env.seed(args.seed + 10)
    LEARN_STEPS = int(args.env.learn_steps)
    agent = make_agent(env, args)

    expert_buffer = []
    expert_policy = []
    for id,(dir,num) in enumerate(zip(args.env.sub_optimal_demo,args.env.num_sub_optimal_demo)):
        add_memory_replay = Memory(1, args.seed)
        add_memory_replay.load(hydra.utils.to_absolute_path(f'experts/{dir}'),
                                num_trajs=num,
                                sample_freq=1,
                                seed=args.seed)
        expert_buffer.append(add_memory_replay)
        print(f'--> Add memory {id} size: {add_memory_replay.size()}')
    
    reward_arr = np.arange(0.1,1.0,1.0/len(args.env.sub_optimal_demo))
    reward_arr[-1] = 1.0
    print(f'rewards for datasets: {reward_arr}')
    
    best_eval_returns = -np.inf
    agent.policy_ls = expert_policy
    learn_steps = 0

    for iter in range(LEARN_STEPS):
        info = {}
        if learn_steps == LEARN_STEPS:
            print('Finished!')
            return
        agent.sqil_update = types.MethodType(sqil_update, agent)
        agent.sqil_update_critic = types.MethodType(sqil_update_critic, agent)
        losses = agent.sqil_update(expert_buffer, learn_steps)
        info.update(losses)
        if learn_steps % args.env.eval_interval == 0:
            eval_returns, eval_timesteps = evaluate(agent, eval_env, num_episodes=args.eval.eps)
            returns = np.mean(eval_returns)
            num_steps = np.mean(eval_timesteps)
            learn_steps += 1  # To prevent repeated eval at timestep 0
            if returns > best_eval_returns:
                best_eval_returns = returns
                print('new best eval',best_eval_returns)
                
            info['Eval/return'] = returns
            info['Eval/std'] = np.std(eval_timesteps)
            info['Eval/best'] = learn_steps
            try:
                wandb.log(info,step = learn_steps)
            except:
                print(info)
        learn_steps += 1
        

              
def sqil_update_critic(self, add_batches,step):
    reward_arr = np.arange(0.1,1.0,1.0/len(add_batches))
    reward_arr[-1] = 1.0
    args = self.args
    batch = concat_data(add_batches,reward_arr, args)
    obs, next_obs, action,reward,done =batch

    with torch.no_grad():
        next_action, log_prob, _ = self.actor.sample(next_obs)
        next_target_Q = self.critic_target(next_obs, next_action)
        next_target_V = next_target_Q  - self.alpha.detach() * log_prob
        target_Q = reward + (1 - done) * self.gamma * next_target_V
    current_Q = self.critic(obs, action)
    current_V = self.getV(obs)
    
    value_loss = (current_V - (1 - done) * self.gamma * next_target_V).mean()
    mse_loss = F.mse_loss(current_Q, target_Q)
    critic_loss = mse_loss + value_loss
    loss_dict  ={
        'value/current_V':current_V.mean().item(),
        'loss/critic_loss':critic_loss.item(),
        'loss/value_loss':value_loss.item(),
        'loss/mse_loss':mse_loss.item(),
    }
    if (step%args.env.eval_interval == 0):
        with torch.no_grad():
            for id,batch in enumerate(add_batches):
                b_obs,b_next_obs,b_action = batch[:3]
                loss_dict[f'log_prob_{id}'] = self.actor.get_log_prob(b_obs, b_action).mean().item()
                loss_dict[f'value/Q_{id}'] = self.critic(b_obs, b_action).mean().item()
        
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()
    return loss_dict

def sqil_update(self, buffers, step):
    add_batches = []
    for id,add_buffer in enumerate(buffers):
        batch = add_buffer.get_samples(self.batch_size, self.device)
        add_batches.append(batch)
    losses = self.sqil_update_critic(add_batches,step)
    
    if self.actor and step % self.actor_update_frequency == 0:
        add_obs = []
        for id,batch in enumerate(add_batches):
            add_obs.append(batch[0])
        add_obs = torch.cat(add_obs,dim=0)
        obs = add_obs
        if self.args.num_actor_updates: 
            for i in range(self.args.num_actor_updates):
                actor_alpha_losses = self.update_actor_and_alpha(obs)
        losses.update(actor_alpha_losses)

    if step % self.critic_target_update_frequency == 0:
        soft_update(self.critic_net, self.critic_target_net, self.critic_tau)
    self.first_log=False
    return losses

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