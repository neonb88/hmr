import matplotlib.pyplot as plt
import numpy as np
np.seterr(all='raise')
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

#######################################################################################################
################################# visualization functions #############################################
#######################################################################################################
#=========================================================================
def pltshow(x):
  plt.imshow(x); plt.show(); plt.close()
#=========================================================================
def cross_sections_biggest(m):
  '''
      @precondition m is a cube (a 3-d np.array)
  '''
  # NOTE:  
  # NOTE:  this isn't working right.  not sure why.  Dec. 12, 2018
  # NOTE:  
  i_3=i_2=i_1=0
  max_vol_3=max_vol_2=max_vol_1=0
  for i in range(m.shape[0]):
      vol_1=np.count_nonzero(m[i,:,:])
      if vol_1 > max_vol_1: max_vol_1 = vol_1; i_1=i
      vol_2=np.count_nonzero(m[:,i,:])
      if vol_2 > max_vol_2: max_vol_2 = vol_2; i_2=i
      vol_3=np.count_nonzero(m[:,:,i])
      if vol_3 > max_vol_3: max_vol_3 = vol_3; i_3=i
  pltshow(m[i_1,:,:])
  pltshow(m[:,i_2,:])
  pltshow(m[:,:,i_3]); return
#=========================================================================
def show_cross_sections(model_3d, axis='z', freq=2):
  # NOTE:   Dec 16, 2018.   Somewhere we're experiencing some weirdness w.r.t. which axis is which.  When this says "z," what we're getting oughta be called either y or x.  when it says "y," the result is fine.  But when it says "x," we get what I thought should be called "z."  I'm not 100% sure how to resolve this.
  if axis.lower()=='z':
      for i in range(model_3d.shape[2]):
          if i%freq==0:
              if np.any(model_3d[:,:,i]):
                  print ('height is {0}   out of {1}'.format(i, model_3d.shape[2]))
                  pltshow(model_3d[:,:,i])
  elif axis.lower()=='y':
      for i in range(model_3d.shape[1]):
          if i%freq==0:
              if np.any(model_3d[:,i,:]):
                  print ('loc is {0}   out of {1}'.format(i, model_3d.shape[1]))
                  pltshow(model_3d[:,i,:])
  elif axis.lower()=='x':
      for i in range(model_3d.shape[0]):
          if i%freq==0:
              if np.any(model_3d[i,:,:]):
                  print ('loc is {0}   out of {1}'.format(i, model_3d.shape[0]))
                  pltshow(model_3d[i,:,:])
  else:
      print ("Usage: please input axis x, y, or z in format:\n\n  show_cross_sections([model_name], axis='z')")
  return
#=========================================================================
def show_all_cross_sections(model_3d, freq=2):
  print ('\n'*3); print ("x: \n\n")
  show_cross_sections(model_3d, axis='x', freq=freq); print ('\n'*3); print ("y: \n\n")
  show_cross_sections(model_3d, axis='y', freq=freq); print ('\n'*3); print ("z: \n\n")
  show_cross_sections(model_3d, axis='z', freq=freq); return
#=========================================================================

# TODO: make a show_3d() function that uses matplotlib
def show_3d(voxels):
  '''
    voxels
  '''
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  x_str='this is the x axis'
  y_str='this is the y axis'
  z_str='this is the z axis'
  ax.set_xlabel(x_str)
  ax.set_ylabel(y_str)
  ax.set_zlabel(z_str)
  ax.voxels(voxels, edgecolor='k')

  plt.show()
  plt.close(); return

#=========================================================================
def plot_skin():
  skin  = np.load("skin_nathan_.npy")
  locs  = np.nonzero(skin)
  print(type(locs))
  fig   = plt.figure()
  ax    = fig.add_subplot(111, projection='3d')
  ax.scatter(locs[0], locs[1], locs[2])
  plt.show()
  plt.close(); return

#=========================================================================
def show_convhull(locs):
  '''
    BUGGY!  Doesn't work; but a correct version of this function is readable HERE (https://stackoverflow.com/questions/27270477/3d-convex-hull-from-point-cloud)

    Curr bugs: (I think we get this b/c of a differences between python2 and python3, but I'm not 100% certain)
      Traceback (most recent call last):
        File "viz.py", line 130, in <module>
          show_convhull(np.load("skin_locs_nathan_.npy"))
        File "viz.py", line 113, in show_convhull
          ax.plot(edges[0],edges[1],edges[2],'bo')
      TypeError: 'zip' object is not subscriptable


    -------
    params:

    Locs.shape is (n,3)
  '''
  import numpy as np
  from mpl_toolkits.mplot3d import Axes3D
  import matplotlib.pyplot as plt
  from scipy.spatial import ConvexHull

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  hull=ConvexHull(locs)

  edges= zip(*locs)

  for i in hull.simplices:
      plt.plot(locs[i,0], locs[i,1], locs[i,2], 'r-')

  ax.plot(edges[0],edges[1],edges[2],'bo')

  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')

  #ax.set_xlim3d(-5,5)
  #ax.set_ylim3d(-5,5)
  #ax.set_zlim3d(-5,5)

  plt.show()
  plt.close();return


if __name__=='__main__':
  #m=np.load('skin_nathan_.npy').astype('bool')
  #show_all_cross_sections(m, freq=20)
  show_convhull(np.load("skin_locs_nathan_.npy"))













































































