import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Observation() :
    def __init__(self) :

        self.states = []
        self.rewards = []
        self.actions = []
        self.advantages = []

class Buffer() :
    def __init__(self):

        self.count = 0
        self.obs = Observation()

    def reset(self) :

        self.count = 0
        self.obs = Observation()
        #maybe invoke garbage collector

    def append(self, s, r, a, ad) :
        #r is vector

        self.count += 1

        self.obs.states.append(s)
        self.obs.rewards.append(r)
        self.obs.actions.append(a)
        self.obs.advantages.append(ad)

    def getSimilarity(self, final_rewards) :

        errors = []
        dim = len(final_rewards)
        for i in range(dim) :
            for j in range(dim) :

                if i == j :
                    continue

                a = final_rewards[i]
                b = final_rewards[j]

                c=a+1j*b
                f = np.angle(c, deg=True)
                g = np.abs(f-45)/45

                errors.append(g)
        sim = 1 - np.mean(np.array(errors))

        return sim

    def plotPath(self, agnumber, ep) :

        fig = plt.figure()
        #print(np.array(self.obs.states))

        for i in range(agnumber) :
            x = np.array(self.obs.states)[:,0+4*i]
            y = np.array(self.obs.states)[:,1+4*i]
            tx = np.array(self.obs.states)[:,2+4*i]
            ty = np.array(self.obs.states)[:,3+4*i]

            re = np.array(self.obs.rewards)[:,i]

            plt.plot(x, y, '-', alpha = 0.7)
            plt.scatter(x,y, c = re)

            plt.scatter(tx, ty, c = 'r')
            plt.text(tx[0], ty[0], str(i), fontsize=12)

            #startpoint
        for i in range(agnumber) :
            x = np.array(self.obs.states)[0,0+4*i]
            y = np.array(self.obs.states)[1,1+4*i]

            lx = np.array(self.obs.states)[-1,0+4*i]
            ly = np.array(self.obs.states)[-1,1+4*i]


            plt.scatter(x,y, c = 'g', s = 200, alpha = 0.4)
            plt.scatter(lx,ly, c = 'r', s = 200, alpha = 0.4)
            plt.text(lx, ly, str(i), fontsize=12)

        plt.axis('off')

        plt.savefig(f'./plots/{ep}.png', bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()

    def plot3d(self, agnumber, ep) :

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(agnumber) :
            x = np.array(self.obs.states)[:,0+6*i]
            y = np.array(self.obs.states)[:,1+6*i]
            z = np.array(self.obs.states)[:,2+6*i]
            tx = np.array(self.obs.states)[:,3+6*i]
            ty = np.array(self.obs.states)[:,4+6*i]
            tz = np.array(self.obs.states)[:,5+6*i]

            re = np.array(self.obs.rewards)[:,i]

            ax.plot(x, y,z, '-', alpha = 0.7)
            #plt.scatter(x,y, c = re)

            ax.scatter(tx, ty, tz, c = 'r')
            #plt.text(tx[0], ty[0], str(i), fontsize=12)

        #plt.axis('off')
        plt.savefig(f'./plots/{ep}.png', bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()

    def saveTraces(self, ep) :
        np.save(f'./traces/states{ep}', self.obs.states)
