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
from torch import nn
import wandb
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter

from make_envs import make_env
from agent import make_agent
from tqdm import tqdm, trange
from dataset.memory import Memory
from utils.utils import eval_mode,evaluate,soft_update,get_concat_samples
from iq_loss import iq_loss
from embed import embed

def get_args(cfg: DictConfig):
    cfg.device = "cuda"
    cfg.hydra_base_dir = os.getcwd()
    return cfg

def compute_energy(embed_1, embed_2):
    squared_diff = torch.square(embed_1.unsqueeze(1) - embed_2.unsqueeze(0))
    energies = torch.sum(-squared_diff, dim=-1)
    return energies

def evaluate(model,dataset,device,args,id):
    _pos_2_3 = []
    _correct_2_3 = []
    _acc = []
    _pred_pos = []
    _pred_correct = []
    _next_pos = []
    _next_correct = []
    
    with torch.no_grad():
        for _ in range(100):
            obs, next_obs, action, _, done = dataset.get_samples(args.embed.eval_batch_size, device)
            next_embed = model.state_function(next_obs)
            obs_embed = model.state_function(obs)
            done = torch.cat((done,1-done),dim=-1)
            latent_action_embed = model.latent_action_function(torch.cat((obs_embed,action),dim=-1))
            
            embed_2 = next_embed
            embed_3 = latent_action_embed
            pred_done = model.done_function(torch.cat((obs_embed,embed_3),dim=-1))
            predicted_labels = (pred_done[:, 1] > pred_done[:, 0]).float()
            true_labels = (done[:, 1] > done[:, 0]).float()
            accuracy = (predicted_labels == true_labels).float().mean()

            energies_2_3 = compute_energy(embed_2, embed_3)
            pos_loss_2_3 = torch.diagonal(energies_2_3, dim1=-2, dim2=-1)
            correct_2_3 = (pos_loss_2_3 >= torch.max(energies_2_3, dim=-1).values).to(torch.float32)
            
            traj_states,traj_actions,traj_next_states = dataset.get_random_traj(device=device)
            _state = traj_states[0].unsqueeze(0)
            latent = model.state_function(_state)
            pred_next_latents = []
            for idx in range(len(traj_states)):
                next_latent = model.latent_action_function(torch.cat((latent,traj_actions[idx].unsqueeze(0)),dim=-1))
                pred_next_latents.append(next_latent)
                latent = next_latent
            pred_next_latents = torch.cat(pred_next_latents,dim=0)
            true_next_latents = model.state_function(traj_next_states)
            pred_energy = compute_energy(pred_next_latents,true_next_latents)
            pos_pred = torch.diagonal(pred_energy, dim1=-2, dim2=-1)
            correct_pred = (pos_pred >= torch.max(pred_energy, dim=-1).values).to(torch.float32)
            
            #---------------------------------------------------------#
            num_future = np.random.randint(2,args.embed.max_horizon + 1)
            state_initial, state_final,action_ls = dataset.get_samples_x_trans(args.embed.eval_batch_size,num_future, device)
            latent_initial = model.state_function(state_initial)
            latent_final = model.state_function(state_final)
            pred_latent = latent_initial
            for idx in range(len(action_ls)-1):
                tmp_action = action_ls[idx]
                pred_latent = model.latent_action_function(torch.cat((pred_latent,tmp_action),dim=-1))
            pred_latent_final = model.latent_action_function(torch.cat((pred_latent,action_ls[-1]),dim=-1))
            
            energies_next = compute_energy(pred_latent_final, latent_final)
            pos_loss_next = torch.diagonal(energies_next, dim1=-2, dim2=-1)
            correct_next = (pos_loss_next >= torch.max(energies_next, dim=-1).values).to(torch.float32)
            
            #---------------------------------------------------------#
            
            _pos_2_3.append(pos_loss_2_3.mean().item())
            _correct_2_3.append(correct_2_3.mean().item())
            _acc.append(accuracy.item())
            _pred_pos.append(pos_pred.mean().item())
            _pred_correct.append(correct_pred.mean().item())
            _next_pos.append(pos_loss_next.mean().item())
            _next_correct.append(correct_next.mean().item())
            
    print(f'[Eval-{id}]:\t done={np.mean(_acc):.2f}| '
        +f'[2-3] P={np.mean(_pos_2_3):.2f}, C={np.mean(_correct_2_3):.2f}\t| '
        +f'[next] P={np.mean(_next_pos):.2f}, C={np.mean(_next_correct):.2f}\t| '
        +f'[rollout] P={np.mean(_pred_pos):.2f}, C={np.mean(_pred_correct):.2f}'
    )

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)
    print(OmegaConf.to_yaml(args))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == 'cuda' and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
    env = make_env(args)
    env.seed(args.seed)
    REPLAY_MEMORY = int(args.env.replay_mem)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    args.embed.obs_dim = obs_dim
    args.embed.action_dim = action_dim
    
    model = embed(obs_dim=obs_dim,action_dim=action_dim,args=args).to(device)
    load_path = f'/home/huy/codes/2023/MLE-IQ/pretrained/noise-{args.env.demo.split(".")[0]}-{args.embed.latent_dim}-{args.embed.max_horizon}'
    try:
        model.load(load_path)
        print(f'load model from {load_path}')
    except:
        print(f'tranning from scratch, save at: {load_path}')
        
    train_memory_replay = Memory(1, args.seed)
    train_memory_replay.load(hydra.utils.to_absolute_path(f'experts/{args.embed.train_dataset}'),
                              num_trajs=args.embed.num_demos,
                              sample_freq=args.expert.subsample_freq,
                              seed=args.seed + 42,save_trajs=True)
    print(f'--> Train memory size: {train_memory_replay.size()}')
    
    random_memory_replay = Memory(1, args.seed)
    random_memory_replay.load(hydra.utils.to_absolute_path(f'experts/{args.embed.random_dataset}'),
                              num_trajs=args.embed.num_demos,
                              sample_freq=args.expert.subsample_freq,
                              seed=args.seed + 42,save_trajs=True)
    print(f'--> Random memory size: {random_memory_replay.size()}')
    
    eval_memory_replay = Memory(1, args.seed)
    eval_memory_replay.load(hydra.utils.to_absolute_path(f'experts/{args.embed.eval_dataset}'),
                              num_trajs=args.embed.num_demos,
                              sample_freq=args.expert.subsample_freq,
                              seed=args.seed + 42,save_trajs=True)
    print(f'--> Eval memory size: {eval_memory_replay.size()}')
    
    arr_pos_1_2 = deque(maxlen=args.embed.log_interval)
    arr_neg_1_2  = deque(maxlen=args.embed.log_interval)
    arr_correct_1_2 = deque(maxlen=args.embed.log_interval)
    arr_pos_1_3 = deque(maxlen=args.embed.log_interval)
    arr_neg_1_3  = deque(maxlen=args.embed.log_interval)
    arr_correct_1_3 = deque(maxlen=args.embed.log_interval)
    arr_pos_2_3 = deque(maxlen=args.embed.log_interval)
    arr_neg_2_3  = deque(maxlen=args.embed.log_interval)
    arr_correct_2_3 = deque(maxlen=args.embed.log_interval)
    arr_pos_next = deque(maxlen=args.embed.log_interval)
    arr_neg_next = deque(maxlen=args.embed.log_interval)
    arr_correct_next = deque(maxlen=args.embed.log_interval)
    
    model.train()
    for iter in range(args.embed.train_steps):
        train_obs, train_next_obs, train_action, _, train_done = \
            train_memory_replay.get_samples(args.embed.batch_size//2, device)
        random_obs, random_next_obs, random_action, _, random_done = \
            random_memory_replay.get_samples(args.embed.batch_size//2, device)
            
        num_future = np.random.randint(2,args.embed.max_horizon + 1)
        train_initial_state, train_final_state,train_action_ls = \
                            train_memory_replay.get_samples_x_trans(args.embed.batch_size//2,num_future, device)
        random_initial_state,random_final_state,random_action_ls = \
                            random_memory_replay.get_samples_x_trans(args.embed.batch_size//2,num_future, device)
            
        obs = torch.cat((train_obs,random_obs),dim=0)
        action = torch.cat((train_action,random_action),dim=0)
        next_obs = torch.cat((train_next_obs,random_next_obs),dim=0)
        _done = torch.cat((train_done,random_done),dim=0)
        done = torch.cat((_done,1-_done),dim=-1)
        
        state_initial = torch.cat((train_initial_state,random_initial_state),dim=0)
        state_final = torch.cat((train_final_state,random_final_state),dim=0)
        action_ls = []
        for (tmp_train_action,tmp_random_action) in zip(train_action_ls,random_action_ls):
            tmp_action = torch.cat((tmp_train_action,tmp_random_action),dim=0)
            action_ls.append(tmp_action)
        #--------------------------------------------------#
        
        this_embed = model.state_action_function(torch.cat((obs,action),dim=-1))
        next_embed = model.state_function(next_obs)
        obs_embed = model.state_function(obs)
        latent_action_embed = model.latent_action_function(torch.cat((obs_embed,action),dim=-1))
        
        if args.embed.shuffle_rate > 0:
            shuffle1 = torch.randperm(this_embed.size(0)).to(device)
            shuffle2 = torch.randperm(next_embed.size(0)).to(device)
            shuffle3 = torch.randperm(latent_action_embed.size(0)).to(device)
            rand1 = torch.rand(this_embed.size(0)).to(device)
            rand2 = torch.rand(next_embed.size(0)).to(device)
            rand3 = torch.rand(latent_action_embed.size(0)).to(device)
            this_embed = torch.where(rand1.unsqueeze(1) < args.embed.shuffle_rate,
                                    this_embed[shuffle1], this_embed)
            next_embed = torch.where(rand2.unsqueeze(1) < args.embed.shuffle_rate,
                                    next_embed[shuffle2], next_embed)
            latent_action_embed = torch.where(rand3.unsqueeze(1) < args.embed.shuffle_rate,
                                    latent_action_embed[shuffle3], latent_action_embed)
            
        embed_1 = this_embed
        embed_2 = next_embed
        embed_3 = latent_action_embed
        pred_done = model.done_function(torch.cat((obs_embed,embed_3),dim=-1))
        
        energies_1_2 = compute_energy(embed_1, embed_2)
        pos_loss_1_2 = torch.diagonal(energies_1_2, dim1=-2, dim2=-1)
        neg_loss_1_2 = torch.logsumexp(energies_1_2, dim=-1)
        model_loss_1_2 = -pos_loss_1_2 + neg_loss_1_2
        correct_1_2 = (pos_loss_1_2 >= torch.max(energies_1_2, dim=-1).values).to(torch.float32)
        
        energies_1_3 = compute_energy(embed_1, embed_3)
        pos_loss_1_3 = torch.diagonal(energies_1_3, dim1=-2, dim2=-1)
        neg_loss_1_3 = torch.logsumexp(energies_1_3, dim=-1)
        model_loss_1_3 = -pos_loss_1_3 + neg_loss_1_3
        correct_1_3 = (pos_loss_1_3 >= torch.max(energies_1_3, dim=-1).values).to(torch.float32)
        
        energies_2_3 = compute_energy(embed_2, embed_3)
        pos_loss_2_3 = torch.diagonal(energies_2_3, dim1=-2, dim2=-1)
        neg_loss_2_3 = torch.logsumexp(energies_2_3, dim=-1)
        model_loss_2_3 = -pos_loss_2_3 + neg_loss_2_3
        correct_2_3 = (pos_loss_2_3 >= torch.max(energies_2_3, dim=-1).values).to(torch.float32)
        
        #--------------------------------------------------#
        with torch.no_grad():
            latent_initial = model.state_function(state_initial)
            latent_final = model.state_function(state_final)
            pred_latent = latent_initial
            for idx in range(len(action_ls)-1):
                tmp_action = action_ls[idx]
                pred_latent = model.latent_action_function(torch.cat((pred_latent,tmp_action),dim=-1))
        pred_latent_final = model.latent_action_function(torch.cat((pred_latent,action_ls[-1]),dim=-1))
        if args.embed.shuffle_rate > 0:
            shuffle1 = torch.randperm(this_embed.size(0)).to(device)
            shuffle2 = torch.randperm(next_embed.size(0)).to(device)
            rand1 = torch.rand(this_embed.size(0)).to(device)
            rand2 = torch.rand(next_embed.size(0)).to(device)
            
            latent_final = torch.where(rand1.unsqueeze(1) < args.embed.shuffle_rate,
                                    latent_final[shuffle1], latent_final)
            pred_latent_final = torch.where(rand2.unsqueeze(1) < args.embed.shuffle_rate,
                                    pred_latent_final[shuffle2], pred_latent_final)
        
        energies_next = compute_energy(pred_latent_final, latent_final)
        pos_loss_next = torch.diagonal(energies_next, dim1=-2, dim2=-1)
        neg_loss_next = torch.logsumexp(energies_next, dim=-1)
        model_loss_next = -pos_loss_next + neg_loss_next
        correct_next = (pos_loss_next >= torch.max(energies_next, dim=-1).values).to(torch.float32)
        
        #--------------------------------------------------#
        model.transition_optimizer.zero_grad()
        BCE_loss = nn.BCELoss()
        done_loss = BCE_loss(pred_done,done)
        loss = model_loss_1_2.mean() + model_loss_1_3.mean() + model_loss_2_3.mean() + model_loss_next.mean() + done_loss
        loss.backward()
        model.transition_optimizer.step()
        model.scheduler.step()
        current_lr = model.scheduler.get_last_lr()[0]
        
        arr_pos_1_2.append(pos_loss_1_2.mean().item())
        arr_neg_1_2.append(neg_loss_1_2.mean().item())
        arr_correct_1_2.append(correct_1_2.mean().item())
        
        arr_pos_1_3.append(pos_loss_1_3.mean().item())
        arr_neg_1_3.append(neg_loss_1_3.mean().item())
        arr_correct_1_3.append(correct_1_3.mean().item())
        
        arr_pos_2_3.append(pos_loss_2_3.mean().item())
        arr_neg_2_3.append(neg_loss_2_3.mean().item())
        arr_correct_2_3.append(correct_2_3.mean().item())
        
        arr_pos_next.append(pos_loss_next.mean().item())
        arr_neg_next.append(neg_loss_next.mean().item())
        arr_correct_next.append(correct_next.mean().item())
        
        if (iter%args.embed.log_interval == 0):
            predicted_labels = (pred_done[:, 1] > pred_done[:, 0]).float()
            true_labels = (done[:, 1] > done[:, 0]).float()
            accuracy = (predicted_labels == true_labels).float().mean()
            print(
                f'[TRAIN] - {iter/args.embed.train_steps*100:.2f}% - lr={current_lr} :\t done={accuracy:.2f}| '
                +f'[2-3] P={np.mean(arr_pos_2_3):.2f}, N={np.mean(arr_neg_2_3):.2f}, C={np.mean(arr_correct_2_3):.2f}\t| '
                +f'[next] P={np.mean(arr_pos_next):.2f}, N={np.mean(arr_neg_next):.2f}, C={np.mean(arr_correct_next):.2f}'
            ,end='\r')
            
        if (iter%args.embed.eval_interval == 0):
            print()
            model.eval()
            evaluate(model=model,dataset=random_memory_replay,device=device,
                     args=args,id='random_memory_replay')
            evaluate(model=model,dataset=train_memory_replay,device=device,
                     args=args,id='train_memory_replay')
            evaluate(model=model,dataset=eval_memory_replay,device=device,
                     args=args,id='eval_memory_replay')
            model.train()
            save_path = \
                f'/home/huy/codes/2023/MLE-IQ/pretrained/noise-{args.env.demo.split(".")[0]}-{args.embed.latent_dim}-{args.embed.max_horizon}'
            print(f'save at: {save_path}')
            model.save(save_path)
            print('-'*20)
    
if __name__ == '__main__':
    main()