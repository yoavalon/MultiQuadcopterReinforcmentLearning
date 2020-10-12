import torch
from torch import nn
from torch.distributions.normal import Normal
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import tracemalloc
from environment import Environment
from storage import Buffer, Observation
import sys

#jeva dz'abi -> schieber

if len(sys.argv)==1:
    agentsNum = 6
else :
    agentsNum = int(sys.argv[1])

writer = SummaryWriter(f'./log/multidiff/{agentsNum}')
tracemalloc.start(5)

env = Environment(False, agentsNum)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(6*env.agentNumber, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

class Actor(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(6*env.agentNumber, 40)
        self.fc2 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, 3*env.agentNumber)
        self.fc4 = nn.Linear(20, 3*env.agentNumber)

        self.n = Normal(torch.tensor([0.]), torch.tensor([0.5]))

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x1 = torch.tanh(self.fc3(x))
        sp = nn.Softplus()
        x2 = sp(self.fc4(x)) #+ 1e-6

        self.n = Normal(x1,x2)
        sample = self.n.rsample()

        return sample

    def getProb(self, s, a):

        #experiment
        x = torch.tanh(self.fc1(s))
        x = torch.tanh(self.fc2(x))
        x1 = torch.tanh(self.fc3(x))
        sp = nn.Softplus()
        x2 = sp(self.fc4(x))

        n = Normal(x1,x2)

        prob = torch.exp(n.log_prob(a))

        return prob

critics = [Critic() for i in range(env.agentNumber)]

actor = Actor()
actor_old = Actor()

actorOptimizer = torch.optim.Adam(actor.parameters(), lr=0.001)
criticOptimizers = [torch.optim.Adam(critic.parameters(), lr=0.001) for critic in critics]

loss_func = torch.nn.MSELoss()

buffer = Buffer()

for ep in range(150000) :
    time1 = tracemalloc.take_snapshot()

    s = env.reset()

    for frame in range(250) :

        st = torch.from_numpy(s.reshape(1,6*env.agentNumber))
        a = np.squeeze(actor(st).detach().numpy())

        next_s, rewards, done, info = env.step(a)

        values = [critic(st) for critic in critics]

        nst = torch.from_numpy(s.reshape(1,6*env.agentNumber))
        next_values = [critic(nst) for critic in critics]

        advantage = np.array((rewards + next_values-values), dtype=np.float32)
        advantage = np.repeat(advantage,3)

        buffer.append(s, rewards, a, advantage)

        s = next_s

        if done :
            break

    if True:
        if ep % 50 == 0 :
            #buffer.plotPath(agentsNum, ep)
            #buffer.plot3d(agentsNum, ep)
            buffer.saveTraces(ep)

        #moved here for slim logs
        writer.add_scalar('actor/ep_length', env.t,ep)
        writer.add_scalar('rewards/r', env.ep_rew, ep)

        for i, ag in enumerate(env.agents) :
            writer.add_scalar(f'rewardsDetail/r{i}', ag.ep_rew, ep)

        sim = buffer.getSimilarity(rewards) #feed with last reward vector
        writer.add_scalar(f'rewards/sim', sim, ep)

        #Critis Training
        for i, critic in enumerate(critics) :
            rews = torch.FloatTensor(np.array(buffer.obs.rewards)[:,i])
            rews = rews.unsqueeze(1)
            vals = critic(torch.FloatTensor(buffer.obs.states))

            cLoss = loss_func(rews, vals)

            writer.add_scalar(f'loss/critic{i}',cLoss,ep)

            criticOptimizers[i].zero_grad

            cLoss.backward()
            [criticOptimizers[i].step() for j in range(1)]

        #Actor Training
        acts = torch.FloatTensor(buffer.obs.actions)
        stats = torch.FloatTensor(buffer.obs.states)

        ratio = actor.getProb(stats, acts)/(actor_old.getProb(stats, acts) + 1e-5)
        adv = torch.tensor(buffer.obs.advantages)
        surr = ratio*adv

        lossActor = -torch.mean(torch.min(surr, torch.clamp(ratio, 0.8,1.2)*adv))
        writer.add_scalar('actor/loss',lossActor,ep)

        actorOptimizer.zero_grad()
        lossActor.backward()
        [actorOptimizer.step() for i in range(1)]

        #copy weights
        actor_old.load_state_dict(actor.state_dict())

        buffer.reset()

    time2 = tracemalloc.take_snapshot()
    mem = tracemalloc.get_traced_memory()[0]
    writer.add_scalar('memory/malloc', mem, ep)
