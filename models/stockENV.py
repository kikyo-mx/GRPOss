import numpy as np
import os
import pandas as pd
import torch
from torch.distributions import Categorical
from stocktrendr import stock_trendr


class StockENV:
    def __init__(self, loader, device='cuda:0'):
        self.loader = loader
        self.hold_num = 0
        self.device = device

    def step(self, action_group, t):
        close = self.loader.dataset[t][1]
        open = self.loader.dataset[t][2]
        reward = (close - open) / open
        advantages = self.advantages(torch.from_numpy(reward[action_group]).to(self.device))
        # s_ = torch.tensor(self.loader.dataset[t + 1][0]).to(self.device)
        return reward, advantages

    def tack_action(self, action_values, topn, g):
        action_groups = []
        for i in range(g):
            action_probs = Categorical(action_values)
            action_group = action_probs.sample([topn]).tolist()
            action_groups.append(action_group)
        return action_groups

    def advantages(self, reward_group):
        mean = torch.mean(reward_group)
        std = torch.std(reward_group)
        advantage = (reward_group - mean) / (std + 1e-4)
        return advantage

def hyedge_list(hg):
    helist = []
    mark = 0
    for i in np.unique(hg[1], return_counts=1)[1]:
        helist.append(hg[0][mark:mark + i].tolist())
    return helist

def get_mask(action_groups, close_trend, st, weight):
    groups_mask = torch.ones([len(action_groups), len(action_groups[0])])
    for i in range(len(action_groups)):
        for j in range(len(action_groups[0])):
            cur_obox = stock_trendr(close_trend[action_groups[i][j]], 0, 30, st)
            if cur_obox and cur_obox[-1][1] >= 29:
                groups_mask[i][j] = weight
    return groups_mask
