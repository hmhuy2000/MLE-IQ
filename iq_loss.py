"""
Copyright 2022 Div Garg. All rights reserved.

Standalone IQ-Learn algorithm. See LICENSE for licensing terms.
"""
import torch
import torch.nn.functional as F
import numpy as np


# Full IQ-Learn objective with other divergences and options
def iq_loss(agent, current_Q, current_v, next_v, batch):
    args = agent.args
    gamma = agent.gamma
    obs, next_obs, action, env_reward, done, is_expert = batch

    loss_dict = {}
    loss_dict.update({
        'value_function/expert_Q':current_Q[is_expert].mean().item(),
        'value_function/pi_Q':current_Q[~is_expert].mean().item(),
        'value_function/expert_v':current_v[is_expert].mean().item(),
        'value_function/pi_v':current_v[~is_expert].mean().item(),
    })
    # keep track of value of initial states
    v0 = agent.getV(obs[is_expert.squeeze(1), ...]).mean()
    loss_dict['value_loss/v0'] = v0.item()

    #  calculate 1st term for IQ loss
    #  -E_(ρ_expert)[Q(s, a) - γV(s')]
    y = (1 - done) * gamma * next_v
    reward = (current_Q - y)[is_expert]
    with torch.no_grad():
        log_pi = agent.actor.get_log_prob(obs=obs,action=action)
        loss_dict.update({
            'update/expert_reward':(current_Q - y)[is_expert].mean().item(),
            'update/pi_reward':(current_Q - y)[~is_expert].mean().item(),
            'update/expert_log_prob':log_pi[is_expert].mean().item(),
            'update/pi_log_prob':log_pi[~is_expert].mean().item(),
        })
    with torch.no_grad():
        # Use different divergence functions (For χ2 divergence we instead add a third bellmann error-like term)
        if args.method.div == "hellinger":
            phi_grad = 1/(1+reward)**2
        elif args.method.div == "kl":
            # original dual form for kl divergence (sub optimal)
            phi_grad = torch.exp(-reward-1)
        elif args.method.div == "kl2":
            # biased dual form for kl divergence
            phi_grad = F.softmax(-reward, dim=0) * reward.shape[0]
        elif args.method.div == "kl_fix":
            # our proposed unbiased form for fixing kl divergence
            phi_grad = torch.exp(-reward)
        elif args.method.div == "js":
            # jensen–shannon
            phi_grad = torch.exp(-reward)/(2 - torch.exp(-reward))
        else:
            phi_grad = 1
    loss = -(phi_grad * reward).mean()
    loss_dict['value_loss/reward_loss'] = loss.item()

    # calculate 2nd term for IQ loss, we show different sampling strategies
    if args.method.loss == "value_expert":
        if (agent.first_log):
            print('value loss: value expert (offline)')
        # sample using only expert states (works offline)
        # E_(ρ)[Q(s,a) - γV(s')]
        value_loss = (current_v - y)[is_expert].mean()
        loss += value_loss
        loss_dict['value_loss/value_loss'] = value_loss.item()

    elif args.method.loss == "value":
        if (agent.first_log):
            print('value loss: value (online)')
        # sample using expert and policy states (works online)
        # E_(ρ)[V(s) - γV(s')]
        value_loss = (current_v - y).mean()
        loss += value_loss
        loss_dict['value_loss/value_loss'] = value_loss.item()
    else:
        raise ValueError(f'This sampling method is not implemented: {args.method.type}')

    if args.method.grad_pen:
        # add a gradient penalty to loss (Wasserstein_1 metric)
        gp_loss = agent.critic_net.grad_pen(obs[is_expert.squeeze(1), ...],
                                            action[is_expert.squeeze(1), ...],
                                            obs[~is_expert.squeeze(1), ...],
                                            action[~is_expert.squeeze(1), ...],
                                            args.method.lambda_gp)
        loss_dict['value_loss/gp_loss'] = gp_loss.item()
        loss += gp_loss

    if args.method.div == "chi" or args.method.chi:  # TODO: Deprecate method.chi argument for method.div
        if (agent.first_log):
            print('regularize loss: chi2 expert (offline)')
        # Use χ2 divergence (calculate the regularization term for IQ loss using expert states) (works offline)
        y = (1 - done) * gamma * next_v

        reward = current_Q - y
        chi2_loss = 1/(4 * args.method.alpha) * (reward**2)[is_expert].mean()
        loss += chi2_loss
        loss_dict['value_loss/chi2_loss'] = chi2_loss.item()

    if args.method.regularize:
        if (agent.first_log):
            print('regularize loss: chi2 (online)')
        # Use χ2 divergence (calculate the regularization term for IQ loss using expert and policy states) (works online)
        y = (1 - done) * gamma * next_v

        reward = current_Q - y
        chi2_loss = 1/(4 * args.method.alpha) * (reward**2).mean()
        loss += chi2_loss
        loss_dict['value_loss/regularize_loss'] = chi2_loss.item()

    loss_dict['value_loss/total_loss'] = loss.item()
    return loss, loss_dict

# Full IQ-Learn objective with other divergences and options
def iq_with_add_loss(agent, current_Q, current_v, next_v, batch,step):
    args = agent.args
    gamma = agent.gamma
    obs, next_obs, action, env_reward, done, is_pi, is_adds = batch

    loss_dict = {}
    loss_dict.update({
        'value_function/pi_Q':current_Q[is_pi].mean().item(),
        'value_function/pi_v':current_v[is_pi].mean().item(),
    })
    for id,is_add in enumerate(is_adds):
        loss_dict.update({
            f'value_function/add_{id}_Q':current_Q[is_add].mean().item(),
            f'value_function/add_{id}_v':current_v[is_add].mean().item(),
        })
    
    # keep track of value of initial states
    v0 = agent.getV(obs[is_adds[-1].squeeze(1), ...]).mean()
    loss_dict['value_loss/v0'] = v0.item()
    y = (1 - done) * gamma * next_v

    pi_reward = (current_Q - y)[is_pi]
    add_rewards = []
    for id,is_add in enumerate(is_adds):
        add_reward = (current_Q - y)[is_add]
        add_rewards.append(add_reward)
    with torch.no_grad():
    #     log_prob = agent.actor.get_log_prob(obs=obs,action=action)
        
        # add_log_probs = []
        # for id,is_add in enumerate(is_adds):
        #     add_log_probs.append(log_prob[is_add].mean().item())
        # max_idx = np.argmax(add_log_probs)
        # for id in range(min(max_idx+1,len(agent.lambd_coefs))):
        #     agent.lambds[id] = agent.lambds[id] + agent.lambd_coefs[id]
        
        C_values = []
        for id in range(len(is_adds)):
            C = 0 #agent.dataset_coefs[id]
            tmp = 0.0
            if (id>0):
                tmp += agent.lambds[id-1]
            if (id<len(is_adds)-1):
                tmp -= agent.lambds[id]
            C += torch.sigmoid(torch.tensor(tmp,dtype=torch.float)).item()
            C_values.append(C)
            loss_dict[f'coefs/C_{id}'] = C
        
        # loss_dict.update({
        #     'log_prob/pi_log_prob':log_prob[is_pi].mean().item(),
        #     'reward/pi_reward':pi_reward.mean().item(),
        # })
        for id,is_add in enumerate(is_adds):
            loss_dict.update({
                # f'log_prob/add_{id}_log_prob':log_prob[is_add].mean().item(),
                f'reward/add_{id}_reward':add_rewards[id].mean().item(),
            })
    loss = 0
    for C,r in zip(C_values,add_rewards):
        loss += -C*r.mean()
    loss_dict['value_loss/reward_loss'] = loss.item()
    # calculate 2nd term for IQ loss, we show different sampling strategies
    if args.method.loss == "value_expert":
        raise NotImplementedError
        if (agent.first_log):
            print('value loss: value expert (offline)')
        # sample using only expert states (works offline)
        # E_(ρ)[Q(s,a) - γV(s')]
        value_loss = (current_v - y)[is_expert].mean()
        loss += value_loss
        loss_dict['value_loss/value_loss'] = value_loss.item()

    elif args.method.loss == "value":
        if (agent.first_log):
            print('value loss: value (online)')
        # sample using expert and policy states (works online)
        # E_(ρ)[V(s) - γV(s')]
        value_loss = (current_v - y).mean()
        loss += value_loss
        loss_dict['value_loss/value_loss'] = value_loss.item()
    else:
        raise ValueError(f'This sampling method is not implemented: {args.method.type}')

    if args.method.div == "chi" or args.method.chi:  # TODO: Deprecate method.chi argument for method.div
        raise NotImplementedError
        if (agent.first_log):
            print('regularize loss: chi2 expert (offline)')
        # Use χ2 divergence (calculate the regularization term for IQ loss using expert states) (works offline)
        y = (1 - done) * gamma * next_v

        reward = current_Q - y
        chi2_loss = 1/(4 * args.method.alpha) * (reward**2)[is_expert].mean()
        loss += chi2_loss
        loss_dict['value_loss/chi2_loss'] = chi2_loss.item()

    if args.method.regularize:
        if (agent.first_log):
            print('regularize loss: chi2 (online)')
        # Use χ2 divergence (calculate the regularization term for IQ loss using expert and policy states) (works online)
        y = (1 - done) * gamma * next_v

        reward = current_Q - y
        chi2_loss = 1/(4 * args.method.alpha) * (reward**2).mean()
        loss += chi2_loss
        loss_dict['value_loss/regularize_loss'] = chi2_loss.item()

    loss_dict['value_loss/total_loss'] = loss.item()
    return loss, loss_dict