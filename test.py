import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

inputs = [torch.randn(1, 36) for _ in range(4)]

lstm1 = nn.LSTM(36,72)
lstm2 = nn.LSTM(72,18)

hidden1 = (torch.randn(1, 1, 72), torch.randn(1, 1, 72))
hidden2 = (torch.randn(1, 1, 18), torch.randn(1, 1, 18))

for i in inputs:
    out1, hidden1 = lstm1(i.view(1, 1, -1), hidden1)
    out2, hidden2 = lstm2(out1.view(1,1,-1), hidden2)

print(out2)



'''
lstm = nn.LSTM(36, 18)  # Input dim is 3, output dim is 3
inputs = [torch.randn(1, 36) for _ in range(4)]  # make a sequence of length 5
hidden = (torch.randn(1, 1, 18), torch.randn(1, 1, 18))
for i in inputs:
    out, hidden = lstm(i.view(1, 1, -1), hidden)
'''
