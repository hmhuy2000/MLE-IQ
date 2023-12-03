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
from dataset.expert_dataset import Traj_dataset
from utils.utils import eval_mode,evaluate,soft_update,concat_data
from torch.utils.data import DataLoader

def get_args(cfg: DictConfig):
    cfg.device = "cuda"
    cfg.hydra_base_dir = os.getcwd()
    return cfg

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)
    
    run_name = f'estimate (Both)'
    for expert_dir,num_expert in zip(args.env.sub_optimal_demo,args.env.num_sub_optimal_demo):
        if ('v3' in args.env.name):
            run_name += f'-{expert_dir.split(".")[0].split("/")[-1]}({int(int(num_expert)/1000)}k)'
        elif ('v2' in args.env.name):
            run_name += f'-{expert_dir.split("-")[0].split("_")[-1]}({int(int(num_expert)/1000)}k)'
            
    print(run_name)
    wandb.init(project=f'test2-{args.env.name}', settings=wandb.Settings(_disable_stats=True), \
        group='offline',
        job_type=run_name,
        name=f'{args.seed}', entity='hmhuy')
    print(OmegaConf.to_yaml(args))
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
    agent.device = device

    from agent.sac_models import RewardFunction
    reward_function = RewardFunction(obs_dim=env.observation_space.shape[0],
                                     action_dim=env.action_space.shape[0],
                                     hidden_dim=args.agent.critic_cfg.hidden_dim,
                                     hidden_depth=args.agent.critic_cfg.hidden_depth,
                                     args=args).to(device)

    reward_optimizer = Adam(reward_function.parameters(),
                                     lr=5e-3)

    expert_buffer = []
    obs_arr = []
    max_size = 0
    for id,(dir,num) in enumerate(zip(args.env.sub_optimal_demo,args.env.num_sub_optimal_demo)):
        add_memory_replay = Memory(1, args.seed)
        obs_arr.append(add_memory_replay.load(hydra.utils.to_absolute_path(f'experts/{dir}'),
                                num_trajs=num,
                                sample_freq=1,
                                seed=args.seed))
        expert_buffer.append(add_memory_replay)
        print(f'--> Add memory {id} size: {add_memory_replay.size()}')
        max_size = max(max_size,len(add_memory_replay.full_trajs['states']))
        
    obs_arr = np.concatenate(obs_arr,axis=0)
    shift = -np.mean(obs_arr, 0)
    scale = 1.0 / (np.std(obs_arr, 0) + 1e-3)
    for buffer in expert_buffer:
        buffer.shift = shift
        buffer.scale = scale
    
    partial_len = 50
    datasets = []
    for id,expert in enumerate(expert_buffer):
        state = expert.full_trajs['states']
        action = expert.full_trajs['actions']
        traj_dataset = Traj_dataset(states=state,actions=action,
                                    shift=shift,scale=scale,
                                    label=id,dup=max_size//len(state),partial_len=partial_len)
        datasets.append(traj_dataset)
    total_iter = 200
    for iter in trange(total_iter+1):
        loaders = []
        for data in datasets:
            loaders.append(DataLoader(data, batch_size=16, shuffle=True))
        for batches in zip(*loaders):
            states_batch,actions_batch,labels_batch = [],[],[]
            for loader_idx, batch in enumerate(batches):
                _states,_actions,_labels = batch
                states_batch.append(_states)
                actions_batch.append(_actions)
                labels_batch.append(_labels)
            states_batch = torch.cat(states_batch,dim=0).to(device)
            actions_batch = torch.cat(actions_batch,dim=0).to(device)
            labels_batch = torch.cat(labels_batch,dim=0).unsqueeze(-1).to(device)
            
            reward = reward_function(states_batch,actions_batch)
            reshaped_reward = reward.view(reward.shape[0], 1, partial_len, 1).expand(-1, partial_len, -1, -1)
            # sum_reward = torch.sum(reward.squeeze(-1),dim=-1,keepdim=True)
            sum_reward = torch.mean(reward.squeeze(-1),dim=-1,keepdim=True)
            pair_wise_sum = sum_reward - sum_reward.view(sum_reward.shape[1],sum_reward.shape[0])
            pair_wise_label = (labels_batch - labels_batch.view(labels_batch.shape[1],labels_batch.shape[0])).clamp(min=-1,max=1)
            max_pair_wise_label = (torch.min(labels_batch,
                        labels_batch.view(labels_batch.shape[1],labels_batch.shape[0]))+1)

            global_loss = (-max_pair_wise_label*pair_wise_sum*pair_wise_label).exp().mean()
            local_loss = torch.square(reshaped_reward - reshaped_reward.permute(0, 2, 1, 3)).mean()
            reward_optimizer.zero_grad()
            (global_loss+local_loss).backward()
            reward_optimizer.step()
        if (iter %100 ==0):
            reward_Q1,reward_Q2,reward_Q3,reward_mean = [],[],[],[]
            for buffer in expert_buffer:
                obs, _, action,_,_ = buffer.get_samples(5000, device)
                rewards = reward_function(obs,action).detach().cpu().numpy()
                reward_Q1.append(int(np.percentile(rewards, 25)*100)/100)
                reward_Q2.append(int(np.percentile(rewards, 50)*100)/100)
                reward_Q3.append(int(np.percentile(rewards, 75)*100)/100)
                reward_mean.append(int(np.mean(rewards)*100)/100)
            print(f'---iter {iter}/{total_iter}:')
            print(f'mean: {reward_mean}')
            print(f'Q1: {reward_Q1}')
            print(f'Q2: {reward_Q2}')
            print(f'Q3: {reward_Q3}')
    reward_arr = reward_mean
    args.expert.reward_arr = reward_arr
    print(f'select reward coefs:')
    print(args.expert.reward_arr)
    best_eval_returns = -np.inf
    best_learn_steps = None
    learn_steps = 0

    for iter in range(LEARN_STEPS):
        info = {}
        if learn_steps == LEARN_STEPS:
            print('Finished!')
            return
        agent.update = types.MethodType(update, agent)
        agent.update_critic = types.MethodType(update_critic, agent)
        losses = agent.update(expert_buffer, learn_steps)
        info.update(losses)
        if learn_steps % args.env.eval_interval == 0:
            eval_returns = evaluate(agent, eval_env,shift,scale, num_episodes=50)
            minimum = np.min(eval_returns)
            maximum = np.max(eval_returns)
            mean_value = np.mean(eval_returns)
            std_value = np.std(eval_returns)            
            Q1 = np.percentile(eval_returns, 25)
            Q2 = np.percentile(eval_returns, 50)
            Q3 = np.percentile(eval_returns, 75)    
            if mean_value > best_eval_returns:
                best_eval_returns = mean_value
                best_learn_steps = learn_steps
                print(f'Best eval: {best_eval_returns:.2f} ± {std_value:.2f}, step={best_learn_steps}')
                        
            info['Eval/mean'] = mean_value
            info['Eval/std'] = std_value
            info['Eval/min'] = minimum
            info['Eval/max'] = maximum
            info['Eval/Q1'] = Q1
            info['Eval/Q2'] = Q2
            info['Eval/Q3'] = Q3
            info['Eval/best_step'] = best_learn_steps
            try:
                wandb.log(info,step = learn_steps)
            except:
                print(info)
        learn_steps += 1
        
def update_critic(self, add_batches,step):
    args = self.args
    reward_arr = args.expert.reward_arr
    if (self.first_log):
        print(f'[reward]: {reward_arr}')
    batch = concat_data(add_batches,reward_arr, args)
    obs, next_obs, action,reward,done =batch

    with torch.no_grad():
        next_action, log_prob, _ = self.actor.sample(next_obs)
        target_next_V = self.critic_target(next_obs, next_action)  - self.alpha.detach() * log_prob
        y_next_V = (1 - done) * self.gamma * target_next_V
        target_Q = reward + y_next_V
    current_Q1,current_Q2 = self.critic(obs, action,both=True)
    current_V = self.getV(obs)
    
    pred_reward_1 = current_Q1 - y_next_V
    pred_reward_2 = current_Q2 - y_next_V
    
    reward_loss_1 = (-reward * pred_reward_1 + 1/2 * (pred_reward_1**2)).mean()
    reward_loss_2 = (-reward * pred_reward_2 + 1/2 * (pred_reward_2**2)).mean()
    reward_loss = (reward_loss_1 + reward_loss_2)/2
    
    if (args.method.loss=='value'):
        if (self.first_log):
            print('[Critic]: use value loss')
        value_loss = (current_V - y_next_V).mean()
    elif (args.method.loss=='v0'):
        if (self.first_log):
            print('[Critic]: use v0 loss')
        value_loss = (1-self.gamma) * current_V.mean()
    else:
        raise NotImplementedError
    
    mse_loss = (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))/2
    critic_loss = mse_loss + value_loss + reward_loss
    loss_dict  ={
        'value/current_V':current_V.mean().item(),
        'loss/critic_loss':critic_loss.item(),
        'loss/value_loss':value_loss.item(),
        'loss/mse_loss':mse_loss.item(),
        'loss/reward_loss':reward_loss.item(),
    }
    if (step%args.env.eval_interval == 0):
        with torch.no_grad():
            for id,batch in enumerate(add_batches):
                b_obs,b_next_obs,b_action,_,b_done = batch
                b_next_action, b_log_prob, _ = self.actor.sample(b_next_obs)
                b_next_target_V = self.critic_target(b_next_obs, b_next_action)  - self.alpha.detach() * b_log_prob
                b_Q1,b_Q2 = self.critic(b_obs, b_action,both=True)
                b_Q = (b_Q1+b_Q2)/2
                b_reward = b_Q - (1 - b_done) * self.gamma * b_next_target_V
                
                pi_action, pi_prob, _ = self.actor.sample(b_obs)
                pi_Q1,pi_Q2 = self.critic(b_obs,pi_action,both=True)
                pi_Q = (pi_Q1+pi_Q2)/2
                pi_reward = pi_Q - (1 - b_done) * self.gamma * b_next_target_V
                
                loss_dict[f'value/pi_Q_{id}'] = pi_Q.mean().item()
                loss_dict[f'reward/pi_reward_{id}'] = pi_reward.mean().item()
                loss_dict[f'log_prob/log_prob_{id}'] = self.actor.get_log_prob(b_obs, b_action).mean().item()
                loss_dict[f'value/Q_{id}'] = b_Q.mean().item()
                loss_dict[f'reward/reward_{id}'] = b_reward.mean().item()
                loss_dict[f'reward/reward_dif_{id}'] = (pi_reward - b_reward).mean().item()
                loss_dict[f'value/Q_dif_{id}'] = (pi_Q - b_Q).mean().item()
        
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()
    return loss_dict

def update(self, buffers, step):
    add_batches = []
    for id,add_buffer in enumerate(buffers):
        batch = add_buffer.get_samples(self.batch_size, self.device)
        add_batches.append(batch)
    losses = self.update_critic(add_batches,step)
    
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