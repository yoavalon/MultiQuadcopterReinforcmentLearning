import numpy as np
import Drone
from panda3d.core import Vec3


class Agent() :

    def __init__(self, factor, visualize) :

        self.visualize = visualize
        self.factor = factor
        self.pos = np.zeros(3)
        self.lastPos = np.zeros(3)
        self.target = np.zeros(3)
        self.ep_rew = 0
        self.done = False

        self.drone = Drone.uav(visualize)

    def getReward(self) :
        diff1 = np.linalg.norm(self.pos-self.target)
        diff2 = np.linalg.norm(self.lastPos-self.target)
        r = diff2 - diff1

        return r

    def getSubState(self) :

        self.pos = self.drone.drone.transform.pos/40

        return np.array([self.pos, self.target], dtype=np.float32).reshape(6,)

    def reset(self) :

        self.ep_rew = 0
        self.pos = np.random.rand(3)-0.5
        self.target = np.random.rand(3)-0.5
        self.done = False
        self.lastPos = self.pos

        self.drone.body.setPos(self.factor*self.pos[0], self.factor*self.pos[1], self.factor*self.pos[2])

        self.drone.body.setHpr(0, 0, 0)
        self.drone.drone.set_linear_velocity(Vec3(0,0,0))
        self.drone.drone.setAngularVelocity(Vec3(0,0,0))

        s = self.getSubState()

        return s

    def step(self, a) :

        basis = np.array([0,0,9.81], dtype = np.float)
        force = 10*a + basis

        force = Vec3(force[0], force[1], force[2])
        self.drone.drone.applyCentralForce(force)

        r = self.getReward()
        s = self.getSubState()

        self.lastPos = np.copy(self.pos)

        self.ep_rew += r

        return s, r , False, {}
