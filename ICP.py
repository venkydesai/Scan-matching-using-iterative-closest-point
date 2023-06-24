import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from numpy.random import normal
from numpy.random import multivariate_normal
from scipy.linalg import expm, sinm, cosm
from scipy.stats import multivariate_normal
from numpy.random import choice

pclX = np.loadtxt("D:/VeNKy/Pycharm Projects/Projects/venv/Mobile Robotics/pclX.txt", dtype=float)
pclY = np.loadtxt("D:/VeNKy/Pycharm Projects/Projects/venv/Mobile Robotics/pclY.txt", dtype=float)


def EstimateCorrespondences(X,Y,t,R,dmax):
  c=[]
  counter=0
  xtemp=[]
  ytemp=[]
  for i in X:
    xi = i.reshape(3, 1)
    k= np.transpose(np.add(np.matmul(R,xi),t))
    # j= np.argmin(np.sum((Y-k)**2,axis=1),axis=0)
    j= np.argmin(np.linalg.norm(Y-k,axis=1),axis=0)
    d= (np.sum((Y[j] -k)**2,axis=1)[0])**0.5
    if d < dmax:
      c.append((counter,j))
      xtemp.append(i)
      ytemp.append(Y[j])
    counter+=1
  xc=np.array(xtemp)
  yc=np.array(ytemp)
  return c,xc,yc

def ComputeOptimalRigidRegistration(xc,yc,C):
  K=len(C)
  x_cent=(np.sum(xc,axis=0)/K).reshape(1,3)
  y_cent=(np.sum(yc,axis=0)/K).reshape(1,3)
  xt= xc-x_cent
  yt= yc-y_cent
  w=np.matmul(np.transpose(yt),xt)/K
  u,sigma, vt =np.linalg.svd(w)
  R= np.matmul(u,vt)
  # temp=(np.matmul(R,np.transpose(x_cent)))
  t=np.transpose(y_cent)-(np.matmul(R,np.transpose(x_cent)))
  return (t,R)

def ICP(X,Y,t,R,dmax,itr):
  for i in range(0,itr):
    C,xc,yc=EstimateCorrespondences(X,Y,t,R,dmax)
    t,R=ComputeOptimalRigidRegistration(xc,yc,C)
  return t,R,xc,yc

R=np.array([[1,0,0],[0,1,0],[0,0,1]])
t= np.array([[0],[0],[0]])
dmax=25
itr=30
t,R,xc,yc=ICP(pclX,pclY,t,R,dmax,itr)
new_X=np.matmul(R,np.transpose(pclX))+t
R=np.array([[1,0,0],[0,1,0],[0,0,1]])
t= np.array([[0],[0],[0]])
dmax=25
itr=30
t,R,xc,yc=ICP(pclX,pclY,t,R,dmax,itr)
new_X=np.matmul(R,np.transpose(pclX))+t

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
X=np.transpose(pclX)
Y=np.transpose(pclY)
ax.scatter(X[0], X[1], X[2], c='blue',s=0.6)
ax.scatter(Y[0], Y[1], Y[2], c='red',s=0.6)
ax.scatter(new_X[0], new_X[1], new_X[2], c='yellow',s=0.6)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
