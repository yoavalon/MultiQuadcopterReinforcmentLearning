from panda3d.core import Vec3
from panda3d.bullet import BulletWorld
from panda3d.bullet import BulletPlaneShape
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletCylinderShape
from panda3d.bullet import ZUp
from panda3d.core import *
#import direct.directbase.DirectStart
import numpy as np

class uav() :

    def __init__(self, visualize) :

        if visualize == False :
            from pandac.PandaModules import loadPrcFileData
            loadPrcFileData("", "window-type none")

        import direct.directbase.DirectStart

        self.initialPos = 10*(np.random.rand(3)-0.5) + 4*(np.random.rand(3)-0.5)
        self.construct()

    def construct(self) :

        # Drone
        name = str(np.random.randint(1,10000))
        self.drone = BulletRigidBodyNode('Box')
        self.drone.setMass(1.0)

        #collision model
        shape2 = BulletBoxShape(Vec3(0.6,1.2,1.45))
        self.drone.addShape(shape2, TransformState.makePos(Point3(0.3, 0.1, -0.1)))
        shape4 = BulletCylinderShape(1, 0.5, ZUp)
        self.drone.addShape(shape4, TransformState.makePos(Point3(0.3, -3.0, 1.0)))
        self.drone.addShape(shape4, TransformState.makePos(Point3(0.3, 3, 1.0)))
        self.drone.addShape(shape4, TransformState.makePos(Point3(3.3, 0.1, 1.0)))
        self.drone.addShape(shape4, TransformState.makePos(Point3(-2.7, 0.1, 1.0)))
        shape5 = BulletBoxShape(Vec3(0.5,3,0.2))
        self.drone.addShape(shape5, TransformState.makePos(Point3(0.3, 0, 1.0)))
        shape6 = BulletBoxShape(Vec3(3,0.5,0.2))
        self.drone.addShape(shape6, TransformState.makePos(Point3(0.3, 0, 1.0)))

        # Drone body
        self.body = render.attachNewNode(self.drone)

        self.body.setPos(self.initialPos[0], self.initialPos[1], self.initialPos[2])
        model = loader.loadModel('./models/drone.gltf')
        model.setHpr(0, 90, 0)
        model.flattenLight()
        model.reparentTo(self.body)

        # blades
        blade = loader.loadModel("./models/blade.gltf")
        blade.reparentTo(self.body)
        blade.setHpr(0, 90, 0)
        blade.setPos(Vec3(0.3, -3.0, 1.1))
        rotation_interval = blade.hprInterval(0.2,Vec3(180,90,0))
        rotation_interval.loop()
        placeholder = self.body.attachNewNode("blade-placeholder")
        placeholder.setPos(Vec3(0, 6.1, 0))
        blade.instanceTo(placeholder)
        placeholder = self.body.attachNewNode("blade-placeholder")
        placeholder.setPos(Vec3(3.05, 3.0, 0))
        blade.instanceTo(placeholder)
        placeholder = self.body.attachNewNode("blade-placeholder")
        placeholder.setPos(Vec3(-3.05, 3.0, 0))
        blade.instanceTo(placeholder)

        #under light
        self.plight2 = PointLight('plight')
        self.plight2.setColor((0.9, 0.1, 0.1, 1))
        plnp = self.body.attachNewNode(self.plight2)
        plnp.setPos(0, 0, -1)
        self.body.setLight(plnp)

        #over light
        plight3 = PointLight('plight')
        plight3.setColor((0.9, 0.8, 0.8, 1))
        plnp = self.body.attachNewNode(plight3)
        plnp.setPos(0, 0, 2)
        self.body.setLight(plnp)
