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
import Drone
import direct.directbase.DirectStart
import sys

if len(sys.argv)==1:
    ep = 50
else :
    ep = int(sys.argv[1])

numAgents = 6
factor = 40
a = np.load(f'./traces/states{ep}.npy')

base.cam.setPos(-50, 50,0)
base.cam.lookAt(0, 0, 0)
base.setBackgroundColor(0.3,0.3,0.7)

wp = WindowProperties()
wp.setSize(1000, 600)
base.win.requestProperties(wp)

world = BulletWorld()
world.setGravity(Vec3(0, 0, -9.81))

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

plight2 = PointLight('plight2')
plight2.setColor((0.1, 0.1, 0.1, 1))
plnp2 = render.attachNewNode(plight2)
plnp2.setPos(0, 0, 20)
render.setLight(plnp2)


# Create Ambient Light
ambientLight = AmbientLight('ambientLight')
#ambientLight.setColor((0.15, 0.05, 0.05, 1))
ambientLight.setColor((0.09, 0.09, 0.15, 1))
ambientLightNP = render.attachNewNode(ambientLight)
render.setLight(ambientLightNP)

wrgDistance = True
positions = []

uavs = [Drone.uav(True) for i in range(numAgents)]

[world.attachRigidBody(uav.drone) for uav in uavs]
[uav.drone.setDeactivationEnabled(False) for uav in uavs]

#lines
lines = LineSegs()
lines.setThickness(2)
lines.setColor((0.3, 0.3, 0.6, 0.1))
node = lines.create()
trace = NodePath(node)
#trace.setColor((0.9, 0.9, 0.9, 0.3))
trace.reparentTo(render)

#extract last position
for i in range(numAgents) :
    x = factor*a[-1][0+6*i]
    y = factor*a[-1][1+6*i]
    z = factor*a[-1][2+6*i]

    uavs[i].body.setPos(x,y, z)

'''
#extract hpr
for i in range(numAgents) :
    x2 = a[-1][0+6*i]
    y2 = a[-1][1+6*i]
    z2 = a[-1][2+6*i]

    x1 = a[-2][0+6*i]
    y1 = a[-2][1+6*i]
    z1 = a[-2][2+6*i]

    direction = factor *np.array([x2-x1, y2-y1, z2-z1])
    uavs[i].body.setHpr(direction[0], direction[1], direction[2])
'''

#extract targets
for i in range(numAgents) :
    tx = factor*a[0][3+6*i]
    ty = factor*a[0][4+6*i]
    tz = factor*a[0][5+6*i]

    targetObj = loader.loadModel("./models/target.gltf")
    targetObj.reparentTo(render)
    targetObj.setPos(Vec3(tx,ty,tz))


for i in range(numAgents) :
    for j in range(a.shape[0]) :
        if j > 0 :
            x0 = factor*a[j-1][0+6*i]
            y0 = factor*a[j-1][1+6*i]
            z0 = factor*a[j-1][2+6*i]

            x = factor*a[j][0+6*i]
            y = factor*a[j][1+6*i]
            z = factor*a[j][2+6*i]

            lines.moveTo(x0,y0,z0)
            lines.drawTo(x,y,z)


        node = lines.create()
        trace = NodePath(node)
        trace.reparentTo(render)


def lightTask(task) :

    count = globalClock.getFrameCount()

    for uav in uavs :

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

def stepTask(task) :

    dt = globalClock.getDt()

    world.doPhysics(0.1)

    return task.cont

#taskMgr.add(stepTask, 'update')
taskMgr.add(lightTask, 'lights')

for i in range(1000000000) :
    taskMgr.step()
