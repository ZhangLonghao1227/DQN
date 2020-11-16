import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, action_dim, device):
        super(DQN, self).__init__()
        self.__conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.__conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.__conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)

        # In DQN nature, we instead the second dense into two denses.
        # self.__fc1 = nn.Linear(64*7*7, 512)
        # self.__fc2 = nn.Linear(512, action_dim)

        self.__fc1 = nn.Linear(64 * 7 * 7, 512)
        self.__fc2_A = nn.Linear(512, action_dim)
        self.__fc2_V = nn.Linear(512, 1)
        self.__fc3 = nn.Linear(512, action_dim)
        self.__device = device

    def forward(self, x):
        x = x / 255.
        x = F.relu(self.__conv1(x))
        x = F.relu(self.__conv2(x))
        x = F.relu(self.__conv3(x))
        x = F.relu(self.__fc1(x.view(x.size(0), -1)))
        x_A = F.relu(self.__fc2_A(x))
        x_V = F.relu(self.__fc2_V(x))

        # update the Q function by formal(2)

        x = x_V + (x_A - x_A.mean(1))
        return self.__fc3(x)