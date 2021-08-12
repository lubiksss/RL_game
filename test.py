import random
from collections import deque as dq
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# Modules needed to transform Mario_env to fit ML Model
from env_transform import *
# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace
# Super Mario environment for OpenAI Gym
import gym_super_mario_bros
# Initialize Super Mario environment
env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
# Limit the action-space to
#   0. walk right
#   1. jump right
env = JoypadSpace(env, [["right"], ["right", "A"]])

# Apply Wrappers to environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
# env = FrameStack(env, num_stack=4)

# NN를 학습시키기 위한 hyperparameter
learning_rate = 0.0005
batch_size = 32
gamma = 0.98
buffer_limit = 100000

gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 강화학습은 Training data set이라는게 따로 없다. Agent가 행동을 취하고 데이터셋을 쌓아나가야합니다.
# 그 데이터셋을 쌓기 위한 버퍼


class ReplayBuffer():
    def __init__(self):
        self.buffer = dq(maxlen=buffer_limit)

    # 버퍼에는 (state, action ,reward, nstate, done) 값이 들어갑니다.
    def put(self, transition):
        self.buffer.append(transition)

    # 샘플 함수를 만드는 이유는 버퍼에 쌓인 데이터셋에서 랜덤으로 학습을 시키기 위함입니다.
    # 그냥 연속해서 쌓인 n개의 데이터셋을 그대로 사용하면 데이터간의 상관관계가 너무 크기 때문에 학슴이 잘 안됩니다.
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float, device=gpu), torch.tensor(a_lst, device=gpu), \
            torch.tensor(r_lst, dtype=torch.float, device=gpu), torch.tensor(s_prime_lst, dtype=torch.float, device=gpu), \
            torch.tensor(done_mask_lst, dtype=torch.float, device=gpu)

    def size(self):
        return len(self.buffer)


memory = ReplayBuffer()
s = env.reset()
a = 0
s_prime, r, done, i = env.step(a)
memory.put((s.unsqueeze(0), a, r/100.0, s_prime.unsqueeze(0), done))

memory.sample(1)
