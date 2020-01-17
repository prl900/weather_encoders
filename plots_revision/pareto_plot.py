import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

z = np.array([0.4148, 0.4549, 0.4923, 0.5710, 0.7944, 0.4356, 0.4675, 0.5695, 0.8360, 0.4516, 0.5305, 0.7217, 1.1311])
y = np.array([0.7090, 0.8355, 0.8438, 0.8570, 0.8833, 0.7075, 0.6815, 0.6563, 0.6247, 0.7997, 0.8197, 0.8328, 0.8444])
x = np.array([0.0278, 0.0550, 0.0553, 0.0584, 0.0678, 0.0277, 0.0231, 0.0200, 0.0161, 0.0433, 0.0466, 0.0490, 0.0503])

z_l = np.array([0.4549, 0.4516, 0.4148, 0.4356, 0.4675, 0.5695, 0.8360])
x_l = np.array([0.0550, 0.0433, 0.0278, 0.0277, 0.0231, 0.0200, 0.0161])

z_r = np.array([0.4549, 0.4148, 0.4356, 0.4675, 0.5695, 0.8360])
y_r = np.array([0.8355, 0.7090, 0.7075, 0.6815, 0.6563, 0.6247])

plt.rcParams.update({'font.size': 16, 'axes.labelweight': 'bold'})
fig= plt.figure()
ax = fig.gca(projection='3d')
#ax.scatter(x,y,z)

ax.plot(x, z, 'ro', zdir='y')#, zs=1.5)
ax.plot(x_l, z_l, 'b-', zdir='y')#, zs=1.5)
ax.plot(y, z, 'go', zdir='x')#, zs=-0.5)
ax.plot(y_r, z_r, 'b-', zdir='x')#, zs=1.5)
ax.plot(x, y, 'ko', zdir='z')#, zs=-1.5)

ax.set_xlabel('POFD', labelpad=13)
ax.set_ylabel('POD', labelpad=13)
ax.set_zlabel('MSE', labelpad=13)
#ax.set_xlim([-0.5, 1.5])
#ax.set_ylim([-0.5, 1.5])
#ax.set_zlim([-1.5, 1.5])

#plt.title('Pareto front between MSE, POD and PFD indices')
plt.show()
