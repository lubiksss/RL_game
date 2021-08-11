import time
import gym
from collections import deque as dq

import torch
import torch.nn as nn
import torch.nn.functional as F


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


env = gym.make('CartPole-v1')
q = Qnet()
q.load_state_dict(torch.load('Cartpole_weight.pt'))

# 주피터 노트북에서 실행 시 env.render()함수가 동작하지 않습니다.
# 시각적으로 보는게 좋을거같아서 .py파일로 만들고 실행시켰습니다.
state = env.reset()
state = torch.tensor(state, dtype=torch.float)
done = False

for i in range(1000+1):

    if done:
        env.render()
        env.step(action)
        time.sleep(0.02)
    else:
        env.render()
        time.sleep(0.01)
        action = q.forward(state).argmax().item()

        state, reward, done, info = env.step(action)
        state = torch.tensor(state, dtype=torch.float)
        print(state, done)

        if done or i == 1000:
            print(i)

env.close()
