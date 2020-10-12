import numpy as np
import gym
from gym.spaces import Discrete, Box
import matplotlib.pyplot as plt
from panda3d.core import Vec3
from panda3d.bullet import BulletWorld
from panda3d.bullet import BulletPlaneShape
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletCylinderShape
from panda3d.bullet import ZUp
from panda3d.core import *
from panda3d.physics import *
from direct.actor.Actor import Actor
from panda3d.bullet import BulletDebugNode
from agent import Agent

class Environment():

    def __init__(self, vis, agentNumber):

        self.visualize = vis
        self.agentNumber = agentNumber
        self.factor = 40

        self.ep_rew = 0
        self.t = 0

        self.agents = [Agent(self.factor, self.visualize) for i in range(self.agentNumber)]

        if self.visualize == False :
            from pandac.PandaModules import loadPrcFileData
            loadPrcFileData("", "window-type none")

        import direct.directbase.DirectStart

        self.construct()
        self.constructAgents()
        self.constructTargets()

        taskMgr.add(self.stepTask, 'update')
        #taskMgr.add(self.lightTask, 'lights')


    def construct(self) :

        if self.visualize :

            base.cam.setPos(60,0,1)
            base.cam.lookAt(0, 0, 0)

            wp = WindowProperties()
            wp.setSize(1200, 500)
            base.win.requestProperties(wp)

        # World
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -9.81))

        #skybox
        skybox = loader.loadModel('./models/skybox.gltf')
        skybox.setTexGen(TextureStage.getDefault(), TexGenAttrib.MWorldPosition)
        skybox.setTexProjector(TextureStage.getDefault(), render, skybox)
        skybox.setTexScale(TextureStage.getDefault(), 1)
        skybox.setScale(100)
        skybox.setHpr(0, -90, 0)

        tex = loader.loadCubeMap('./textures/s_#.jpg')
        skybox.setTexture(tex)
        skybox.reparentTo(render)

        render.setTwoSided(True)

        #Light
        plight = PointLight('plight')
        plight.setColor((1.0, 1.0, 1.0, 1))
        plnp = render.attachNewNode(plight)
        plnp.setPos(0, 0, 0)
        render.setLight(plnp)

        # Create Ambient Light
        ambientLight = AmbientLight('ambientLight')
        ambientLight.setColor((0.15, 0.05, 0.05, 1))
        ambientLightNP = render.attachNewNode(ambientLight)
        render.setLight(ambientLightNP)

    def constructAgents(self) :

        [self.world.attachRigidBody(agent.drone.drone) for agent in self.agents]
        [agent.drone.drone.setDeactivationEnabled(False) for agent in self.agents]

    def constructTargets(self) :

        self.targets = []
        for i, agent in enumerate(self.agents) :
            self.targets.append(loader.loadModel("./models/target.gltf"))
            self.targets[i].reparentTo(render)
            self.targets[i].setPos(Vec3(self.factor*agent.target[0], self.factor*agent.target[1], self.factor*agent.target[2]))

    def resetTargets(self) :
        for i, agent in enumerate(self.agents) :
            self.targets[i].setPos(Vec3(self.factor*agent.target[0], self.factor*agent.target[1], self.factor*agent.target[2]))


    def lightTask(self, task) :

        count = globalClock.getFrameCount()

        for uav in self.uavs :

            rest = count % 100
            if rest < 10 :
                uav.plight2.setColor((0.1, 0.9, 0.1, 1))
            elif rest > 30 and rest < 40 :
                uav.plight2.setColor((0.9, 0.1, 0.1, 1))
            elif rest > 65 and rest < 70 :
                uav.plight2.setColor((0.9,0.9, 0.9, 1))
            elif rest > 75 and rest < 80 :
                uav.plight2.setColor((0.9,0.9, 0.9, 1))
            else :
                uav.plight2.setColor((0.1, 0.1, 0.1, 1))

        return task.cont

    def stepTask(self, task) :

        dt = globalClock.getDt()

        if self.visualize :
            self.world.doPhysics(dt)
        else :
            self.world.doPhysics(0.1)

        return task.cont

    def getState(self) :
        state = [ag.getSubState() for ag in self.agents]

        return np.array(state).reshape(-1)

    def getReward(self) :

        rewards = [ag.getReward() for ag in self.agents]
        return np.array(rewards)

    def reset(self):

        self.ep_rew = 0
        rewards = [ag.reset() for ag in self.agents]
        self.t = 0
        state = self.getState()
        self.resetTargets()

        return state

    def step(self, a) :

        done = False
        self.t +=1

        #apply action
        for i, ag in enumerate(self.agents) :
            subAction = a[(3*i):(3*(i+1))]
            ag.step(subAction) #-> in agent apply force !!

        taskMgr.step()

        '''
        #collision test
        if self.world.contactTestPair(self.uavs[0].drone, self.uavs[1].drone).getNumContacts() > 0 :
            print('collision')
            #done = True
        '''

        state = self.getState()
        rewards = self.getReward()

        self.ep_rew += np.mean(rewards)

        '''
        dones = [ag.done for ag in self.agents]

        if any(dones) :     #changed from all
            done = True
        '''

        #print(self.t)

        if self.t > 200 :
            done = True

        return state, rewards, done, {}
