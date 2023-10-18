import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import hydra
import utils.utils as utils

class embed(nn.Module):
    def __init__(self, obs_dim, action_dim, args):    
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.args = args
        self.embed_args = args.embed
        self.device = torch.device(args.device)
        self.state_action_function = utils.mlp(obs_dim + action_dim, self.embed_args.hidden_dim,
                                self.embed_args.latent_dim, self.embed_args.hidden_depth)
        self.state_function = utils.mlp(obs_dim , self.embed_args.hidden_dim,
                                self.embed_args.latent_dim, self.embed_args.hidden_depth)
        self.latent_action_function = utils.mlp(self.embed_args.latent_dim + action_dim , self.embed_args.hidden_dim,
                                self.embed_args.latent_dim, self.embed_args.hidden_depth)
        self.done_function = utils.mlp(self.embed_args.latent_dim + self.embed_args.latent_dim ,
                                       self.embed_args.hidden_dim,2, self.embed_args.hidden_depth,
                                       output_mod=nn.Sigmoid())
        self.transition_optimizer = Adam([
                                    {'params': self.state_action_function.parameters()},
                                    {'params': self.state_function.parameters()},
                                    {'params': self.latent_action_function.parameters()},
                                    {'params': self.done_function.parameters()},                                    
                                    ], lr=args.embed.lr)
        
    def save(self,save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.state_action_function.state_dict(),f'{save_path}/state_action_function.pth')
        torch.save(self.state_function.state_dict(),f'{save_path}/state_function.pth')
        torch.save(self.latent_action_function.state_dict(),f'{save_path}/latent_action_function.pth')
        torch.save(self.done_function.state_dict(),f'{save_path}/done_function.pth')
        
    def load(self,load_path):
        if (not os.path.exists(load_path)):
            raise
        self.state_action_function.load_state_dict(torch.load(
            f'{load_path}/state_action_function.pth'))
        self.state_function.load_state_dict(torch.load(
            f'{load_path}/state_function.pth'))
        self.latent_action_function.load_state_dict(torch.load(
            f'{load_path}/latent_action_function.pth'))
        self.done_function.load_state_dict(torch.load(
            f'{load_path}/done_function.pth'))