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
from torch import nn
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter
from torch.optim import Adam
from make_envs import make_env
from agent import make_agent
from dataset.memory import Memory
from utils.utils import eval_mode,evaluate,soft_update,get_concat_with_add_samples
from iq_loss import iq_with_add_loss
from BC_policies.actor import StateIndependentPolicy

def get_args(cfg: DictConfig):
    cfg.device = "cuda"
    cfg.hydra_base_dir = os.getcwd()
    return cfg

def train_BC(policy,dataset,device,target_log_prob,save_path,total_steps):
    optim_actor = Adam(policy.parameters(), lr=1e-4)
    arr_log = deque(maxlen=1000)
    from tqdm import tqdm
    pbar = tqdm(range(total_steps))
    for iter in pbar:
        policy_batch = dataset.get_samples(256, device)
        obs, next_obs, action = policy_batch[:3]
        log_prob = policy.evaluate_log_pi(obs,action)
        loss = -target_log_prob*log_prob.mean() + 1/2 *(log_prob**2).mean()
        arr_log.append(log_prob.mean().item())
        optim_actor.zero_grad()
        loss.backward(retain_graph=False)
        optim_actor.step()
        if (iter%1000 == 0):
            pbar.set_description(f'{np.mean(arr_log):.3f}')
    torch.save(policy.state_dict(), save_path)
    return policy

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)
    
    run_name = f'new-MLE'
    for expert_dir,num_expert in zip(args.env.sub_optimal_demo,args.env.num_sub_optimal_demo):
        run_name += f'-{expert_dir.split(".")[0].split("/")[-1]}({num_expert})'
    wandb.init(project='MLE-mujoco', settings=wandb.Settings(_disable_stats=True), \
        group=args.env.name,
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
    REPLAY_MEMORY = int(args.env.replay_mem)
    INITIAL_MEMORY = int(args.env.initial_mem)
    EPISODE_STEPS = int(args.env.eps_steps)
    EPISODE_WINDOW = int(args.env.eps_window)
    LEARN_STEPS = int(args.env.learn_steps)
    INITIAL_STATES = 128
    agent = make_agent(env, args)

    expert_buffer = []
    expert_policy = []
    for id,(dir,num) in enumerate(zip(args.env.sub_optimal_demo,args.env.num_sub_optimal_demo)):
        add_memory_replay = Memory(1, args.seed)
        add_memory_replay.load(hydra.utils.to_absolute_path(f'experts/{dir}'),
                                num_trajs=num,
                                sample_freq=1,
                                seed=args.seed + id*43)
        expert_buffer.append(add_memory_replay)
        policy_path = f'/home/huy/codes/2023/MLE_codes/MLE-IQ/BC_policies/{dir.split(".")[0]}_{num}.pth'
        policy = StateIndependentPolicy(
                state_shape=env.observation_space.shape,
                action_shape=env.action_space.shape,
                hidden_units=[256,256],
                hidden_activation=nn.Tanh(),
            ).to(device)
        if (num>=50 and os.path.isfile(policy_path)):
            policy.load_state_dict(torch.load(policy_path))
        else:
            print(f'train BC for {policy_path}')
            if (num>= 50):
                total_steps=int(3e5)
            else:
                total_steps=int(1e5)
            os.makedirs(f'/home/huy/codes/2023/MLE_codes/MLE-IQ/BC_policies/{dir.split("/")[0]}',exist_ok=True)
            policy = train_BC(policy=policy,dataset=add_memory_replay,device=device,
                              target_log_prob=env.action_space.shape[0],save_path=policy_path,
                              total_steps=total_steps)
        policy.eval()
        expert_policy.append(policy)
        policy_batch = add_memory_replay.get_samples(1000, device)
        obs, next_obs, action = policy_batch[:3]
        log_prob = policy.evaluate_log_pi(obs,action)
        print(f'--> Add memory {id} size: {add_memory_replay.size()}')
        print(f'--> policy {id} log_prob: {log_prob.mean().item():.3f}')
        print()
    online_memory_replay = Memory(REPLAY_MEMORY//2, args.seed+1)
    steps = 0
    steps_window = deque(maxlen=EPISODE_WINDOW) 
    rewards_window = deque(maxlen=EPISODE_WINDOW)
    best_eval_returns = -np.inf
    agent.policy_ls = expert_policy
    learn_steps = 0
    episode_reward = 0
    
    for epoch in count():
        state = env.reset()
        episode_reward = 0
        done = False
        for episode_step in range(EPISODE_STEPS):
            info = {}
            with eval_mode(agent):
                action = agent.choose_action(state, sample=True)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            done_no_lim = done
            if str(env.__class__.__name__).find('TimeLimit') >= 0 and episode_step + 1 == env._max_episode_steps:
                done_no_lim = 0
            online_memory_replay.add((state, next_state, action, reward, done_no_lim))
            
            if online_memory_replay.size() > INITIAL_MEMORY:
                learn_steps += 1
                if learn_steps == LEARN_STEPS:
                    print('Finished!')
                    return

                ######
                # IQ-Learn Modification
                agent.iq_update = types.MethodType(iq_update, agent)
                agent.iq_update_critic = types.MethodType(iq_update_critic, agent)
                losses = agent.iq_update(online_memory_replay,
                                         expert_buffer, learn_steps)
                info.update(losses)
                ######

            if learn_steps % args.env.eval_interval == 0:
                eval_returns, eval_timesteps = evaluate(agent, eval_env, num_episodes=args.eval.eps)
                returns = np.mean(eval_returns)
                num_steps = np.mean(eval_timesteps)
                learn_steps += 1  # To prevent repeated eval at timestep 0
                print('EVAL\tEp {}\tAverage reward: {:.2f}\t'.format(epoch, returns))

                if returns > best_eval_returns:
                    best_eval_returns = returns
                    save(agent, epoch, args, output_dir='results_best')
                info['Eval/return'] = returns
                info['Eval/num_steps'] = num_steps
                info['Train/return'] = np.mean(rewards_window)
                info['Train/num_steps'] = np.mean(steps_window)
                try:
                    wandb.log(info,step = learn_steps)
                except:
                    print(info)
            elif learn_steps % args.env.log_interval == 0:
                info['Train/return'] = np.mean(rewards_window)
                info['Train/num_steps'] = np.mean(steps_window)
                try:
                    wandb.log(info,step = learn_steps)
                except:
                    print(info)

            if (done):
                break
            state = next_state
        rewards_window.append(episode_reward)
        steps_window.append(episode_step)
        print('TRAIN\tEp {}\tAverage reward: {:.2f}\t'.format(epoch, np.mean(rewards_window)))
        save(agent, epoch, args, output_dir='results')
              
def iq_update_critic(self, policy_batch, add_batches,step):
    args = self.args
    batch = get_concat_with_add_samples(policy_batch, add_batches, args)
    obs, next_obs, action = batch[0:3]

    agent = self
    current_V = self.getV(obs)
    if args.train.use_target:
        with torch.no_grad():
            next_V = self.get_targetV(next_obs)
    else:
        next_V = self.getV(next_obs)

    if "DoubleQ" in self.args.q_net._target_:
        raise NotImplementedError
    else:
        current_Q = self.critic(obs, action)
        critic_loss, loss_dict = iq_with_add_loss(agent, current_Q, current_V, next_V, batch,step)

    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()
    return loss_dict

def iq_update(self, policy_buffer, buffers, step):
    policy_batch = policy_buffer.get_samples(self.batch_size, self.device)
    add_batches = []
    for id,add_buffer in enumerate(buffers):
        batch = add_buffer.get_samples(self.batch_size, self.device)
        add_batches.append(batch)
    losses = self.iq_update_critic(policy_batch ,add_batches,step)
    
    with torch.no_grad():
        new_add_log_probs = []
        obs = policy_batch[0]
        agent_actions = self.actor(obs).sample()
        for id,policy in enumerate(self.policy_ls):
            new_log_prob = policy.evaluate_log_pi(obs,agent_actions)
            new_add_log_probs.append(new_log_prob.mean().item())
            losses.update({
                f'log_prob/add_{id}_log_prob':new_log_prob.mean().item(),
            })
        max_idx = np.argmax(new_add_log_probs)
        for id in range(min(max_idx+1,len(self.lambd_coefs))):
            self.lambds[id] = self.lambds[id] + self.lambd_coefs[id]
        losses.update({
            'update/max_idx':max_idx,
        })
        
    if self.actor and step % self.actor_update_frequency == 0:
        if self.args.offline:
            if (self.first_log):
                print('actor training: only data (offline)')
            raise NotImplementedError
        else:
            if (self.first_log):
                print('actor training: both data (online)')
            add_obs = []
            for id,batch in enumerate(add_batches):
                if (id<max_idx):
                    continue
                add_obs.append(batch[0])
            add_obs = torch.cat(add_obs,dim=0)
            obs = torch.cat([policy_batch[0], add_obs], dim=0)
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