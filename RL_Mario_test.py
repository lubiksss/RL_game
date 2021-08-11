import time
import gym
from collections import deque as dq
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(55632, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.to(device)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 55632)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.float()


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, [['right'], ['right', 'A']])

q = Qnet().to(device)
q.load_state_dict(torch.load('Mario_weight.pt'))

# 주피터 노트북에서 실행 시 env.render()함수가 동작하지 않습니다.
# 시각적으로 보는게 좋을거같아서 .py파일로 만들고 실행시켰습니다.
state = env.reset()
state = Image.fromarray(state).convert('L')
state = np.array(state, 'uint8')
state = state.reshape((1,)+state.shape)
state = state.copy()
done = False

for i in range(1000+1):

    env.render()
    time.sleep(0.01)
    action = q.forward(torch.from_numpy(
        state.reshape((1,)+state.shape)).float()).argmax().item()

    state, reward, done, info = env.step(action)
    state = Image.fromarray(state).convert('L')
    state = np.array(state, 'uint8')
    state = state.reshape((1,)+state.shape)
    state = state.copy()

    if done or i == 1000:
        print(i)

env.close()
