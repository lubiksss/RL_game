import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.image as img
from torchvision import transforms as T
import matplotlib.pyplot as plt

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros

env = gym_super_mario_bros.make('SuperMarioBros-v0')
# env = gym_super_mario_bros.make('SuperMarioBros-8-4-v0')
# Limit the action-space to
#   0. walk right
#   1. jump right
env = JoypadSpace(env, [['right'], ['right', 'A']])

test = env.reset()
# permute [H, W, C] array to [C, H, W] tensor
test = np.transpose(test, (2, 0, 1))
test = torch.tensor(test.copy(), dtype=torch.float)
transform = T.Grayscale()
test = transform(test)

plt.imshow(test.permute(1, 2, 0))
plt.show()
