from env_transform import *
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

gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Mario의 state가 (4, 84, 84)sahpe, action은 2개이기 때문에 input 3, output 2인 CNN생성
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(5184, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = x.to(gpu)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 5184)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.float()


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, [['right'], ['right', 'A']])

q = Qnet().to(gpu)
q.load_state_dict(torch.load('Mario_weight.pt'))

# 주피터 노트북에서 실행 시 env.render()함수가 동작하지 않습니다.
# 시각적으로 보는게 좋을거같아서 .py파일로 만들고 실행시켰습니다.
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
state = env.reset()

for i in range(1000+1):

    env.render()
    action = q.forward(state.unsqueeze(0).unsqueeze(0)).argmax().item()

    state, _, _, _ = env.step(action)
    time.sleep(0.01)


env.close()
