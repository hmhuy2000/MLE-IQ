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
from agent import make_agent
from dataset.memory import Memory
from utils.utils import eval_mode,evaluate,soft_update,get_concat_samples
from iq_loss import iq_loss

def get_args(cfg: DictConfig):
    cfg.device = "cuda"
    cfg.hydra_base_dir = os.getcwd()
    return cfg

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)
    
    wandb.init(project='MLE-mujoco', settings=wandb.Settings(_disable_stats=True), \
        group=args.env.name,
        job_type=f'IQ-{args.env.demo.split(".")[0].split("/")[-1]}({args.expert.demos})',
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

    # Load expert data
    expert_memory_replay = Memory(REPLAY_MEMORY//2, args.seed)
    expert_memory_replay.load(hydra.utils.to_absolute_path(f'experts/{args.env.demo}'),
                              num_trajs=args.expert.demos,
                              sample_freq=args.expert.subsample_freq,
                              seed=args.seed + 42)
    print(f'--> Expert memory size: {expert_memory_replay.size()}')
    online_memory_replay = Memory(REPLAY_MEMORY//2, args.seed+1)
    
    steps = 0

    steps_window = deque(maxlen=EPISODE_WINDOW) 
    rewards_window = deque(maxlen=EPISODE_WINDOW)
    best_eval_returns = -np.inf

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
                                         expert_memory_replay, learn_steps)
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
                wandb.log(info,step = learn_steps)
            elif learn_steps % args.env.log_interval == 0:
                info['Train/return'] = np.mean(rewards_window)
                info['Train/num_steps'] = np.mean(steps_window)
                wandb.log(info,step = learn_steps)

            if (done):
                break
            state = next_state
        rewards_window.append(episode_reward)
        steps_window.append(episode_step)
        print('TRAIN\tEp {}\tAverage reward: {:.2f}\t'.format(epoch, np.mean(rewards_window)))
        save(agent, epoch, args, output_dir='results')
            
   
def iq_update_critic(self, policy_batch, expert_batch):
    args = self.args
    policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = policy_batch
    expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch

    if args.only_expert_states:
        # Use policy actions instead of experts actions for IL with only observations
        expert_batch = expert_obs, expert_next_obs, policy_action, expert_reward, expert_done

    batch = get_concat_samples(policy_batch, expert_batch, args)
    obs, next_obs, action = batch[0:3]

    agent = self
    current_V = self.getV(obs)
    if args.train.use_target:
        with torch.no_grad():
            next_V = self.get_targetV(next_obs)
    else:
        next_V = self.getV(next_obs)

    if "DoubleQ" in self.args.q_net._target_:
        current_Q1, current_Q2 = self.critic(obs, action, both=True)
        q1_loss, loss_dict1 = iq_loss(agent, current_Q1, current_V, next_V, batch)
        q2_loss, loss_dict2 = iq_loss(agent, current_Q2, current_V, next_V, batch)
        critic_loss = 1/2 * (q1_loss + q2_loss)
        # merge loss dicts
        raise
    else:
        current_Q = self.critic(obs, action)
        critic_loss, loss_dict = iq_loss(agent, current_Q, current_V, next_V, batch)

    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()
    return loss_dict

def iq_update(self, policy_buffer, expert_buffer, step):
    policy_batch = policy_buffer.get_samples(self.batch_size, self.device)
    expert_batch = expert_buffer.get_samples(self.batch_size, self.device)

    losses = self.iq_update_critic(policy_batch, expert_batch)
    if self.actor and step % self.actor_update_frequency == 0:
        if not self.args.agent.vdice_actor:
            if self.args.offline:
                if (self.first_log):
                    print('actor training: only data (offline)')
                obs = expert_batch[0]
            else:
                if (self.first_log):
                    print('actor training: both data (online)')
                obs = torch.cat([policy_batch[0], expert_batch[0]], dim=0)

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