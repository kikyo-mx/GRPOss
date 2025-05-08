import time
import random
import torch
from torch_geometric import nn
import torch.nn.functional as F
from torch import optim
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple       # 队列类型
from tqdm import tqdm
from stocktrendr import stock_trendr
from models.transformer import FeatureExtractor


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))



# ----------------------------------------- #
# 策略网络
# ----------------------------------------- #


class PolicyNet(torch.nn.Module):
    def __init__(self, n_observations, n_actions):
        super(PolicyNet, self).__init__()
        self.gru = torch.nn.GRU(input_size=n_observations, hidden_size=32, batch_first=True)
        self.lstm = torch.nn.LSTM(input_size=n_observations, hidden_size=32, batch_first=True)
        self.transformer = FeatureExtractor(n_observations, d_model=32)
        # self.Attn = ProbAttention
        self.drop = torch.nn.Dropout(p=0.2)

        self.hatt1 = nn.HypergraphConv(32, 32, use_attention=False, heads=1, concat=False,
                                       negative_slope=0.2, dropout=0.5, bias=True)
        self.hatt2 = nn.HypergraphConv(32, 32, use_attention=True, heads=1, concat=False,
                                       negative_slope=0.2, dropout=0.5, bias=True)

        self.linear1 = torch.nn.Linear(32, 32)
        self.linear2 = torch.nn.Linear(32, n_actions)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, e, weight):
        x = self.transformer(x.float())
        x = F.relu(x)
        # x = self.hatt1(x, e)
        x = self.linear1(x)
        x = self.linear2(x).squeeze(0)
        x = F.softmax(x.squeeze(1), dim=0)
        return x


# ----------------------------------------- #
# 经验回放池
# ----------------------------------------- #


class ReplayMemory(object):
    def __init__(self, memory_size, device='cuda:0'):
        self.memory = deque([], maxlen=memory_size)
        self.weights = deque([], maxlen=memory_size)
        self.device = device

    def sample(self, batch_size):
        batch = Transition(*zip(*random.choices(self.memory, k=batch_size, weights=self.weights)))
        state = torch.cat(batch.state).to(self.device)
        batch_action = torch.tensor(batch.action).to(self.device)
        batch_reward = torch.tensor(batch.reward).to(self.device)
        # batch_action = torch.cat(batch.action)
        # batch_reward = torch.cat(batch.reward)
        next_state = torch.cat(batch.next_state).to(self.device)

        return state, batch_action, batch_reward, next_state

    def push(self, batch, weight):
        self.memory.append(batch)
        self.weights.append(weight)


    def __len__(self):
        return len(self.memory)


# ----------------------------------------- #
# GRPO构建
# ----------------------------------------- #


class GRPOLoss(torch.nn.Module):
    def __init__(self, clip_eps: float = 0.2, kl_weight: float = 0.01, device='cuda:1'):
        super().__init__()
        self.clip_eps = clip_eps  # 策略更新的裁剪阈值
        self.kl_weight = kl_weight  # KL散度的权重系数
        self.device = device

    def compute_advantages(self, rewards):
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        return advantages

    def compute_loss(
            self,
            old_logprobs: torch.Tensor,  # 旧策略的对数概率 [batch_size]
            new_logprobs: torch.Tensor,  # 新策略的对数概率 [batch_size]
            ref_logprobs: torch.Tensor,  # 参考的对数概率 [batch_size]
            normalized_advantages: torch.Tensor,  # 分组归一化后的优势值 [batch_size]
            group_mask: torch.Tensor=None,  # 分组掩码 [batch_size, num_groups]
    ) -> torch.Tensor:
        """
        计算 GRPO 总损失，包含策略损失、KL散度约束
        """
        loss = torch.zeros(old_logprobs.shape[0]).to(self.device)
        for i in range(old_logprobs.shape[0]):

            # --- 策略损失计算（含分组相对优势）---
            ratio = torch.exp(new_logprobs[i] - old_logprobs[i].detach())  # 重要性采样比率
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)

            # 策略损失（含裁剪约束）
            policy_loss = -torch.min(ratio * normalized_advantages[i], clipped_ratio * normalized_advantages[i])

            # --- KL散度计算（新旧策略差异约束）---
            kl_div = torch.exp(ref_logprobs[i] - new_logprobs[i]) - (ref_logprobs[i] - new_logprobs[i]) - 1

            # --- 总损失 ---
            total_loss = policy_loss + self.kl_weight * kl_div

            # loss[i] = (total_loss * group_mask[i]).mean()
            loss[i] = (total_loss * group_mask[i]).sum() / group_mask[i].sum()

        return loss

