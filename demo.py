"""
Demo of HMR.

Note that HMR requires the bounding box of the person in the image. The best performance is obtained when max length of the person in the image is roughly 150px. 

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out the bbox and the right scale factor.

Sample usage:

# On images on a tightly cropped image around the person
python -m demo --img_path data/im1963.jpg
python -m demo --img_path data/coco1.png

# On images, with openpose output
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
"""
from __future__ import absolute_import # TODO: read up on this.  Is there a better relative_import one could implement?
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np
import subprocess as sp
import os
import time

import skimage.io as io
import tensorflow as tf

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel
#import matplotlib as mpl
#mpl.use('Agg') # 'TkAgg'
import matplotlib.pyplot as plt

config = flags.FLAGS
flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')

outmesh_path = '/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/src/web/upload_img_flask/mesh.obj' # faces in src/util/renderer.py   .   We should really learn how to use absl and configs and standardize

#=========================================================================
def pe(n=89): print('='*n)
#=========================================================================
def pltshow(x):
  plt.imshow(x); plt.show(); plt.close()
#=========================================================================
pr=print
def pe(n=89):
  print("="*n)
def pn(n=0):
  print("\n"*n)
#=========================================================================
def fix():
  # flips .obj file so faces are after vertices
  fresh_outmesh_path  = '/home/n/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/src/web/upload_img_flask/freshmesh.obj'
  # TODO: fix this everywhere by storing the proper filename  in a different .txt file (ie. 'obj_filename.txt')
  with open(outmesh_path, 'r') as fp:
    lines=fp.readlines()
  with open(fresh_outmesh_path, 'w') as fp:
    for line in lines:
      if 'v' in line:
        fp.write(line)
    for line in lines:
      if 'f' in line:
        fp.write(line)
  sp.call(['mv', '-f', fresh_outmesh_path, outmesh_path])
#=========================================================================
def visualize(img, proc_param, joints, verts, cam):
    """
    Renders the result in original image coordinate frame.
    """
    funcname=  sys._getframe().f_code.co_name
    print("entering function "+funcname)
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    # Render results
    skel_img = vis_util.draw_skeleton(img, joints_orig)
    rend_img_overlay = renderer(
        vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
    rend_img = renderer(
        vert_shifted, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp1 = renderer.rotated(
        vert_shifted, 60, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp2 = renderer.rotated(
        vert_shifted, -60, cam=cam_for_render, img_size=img.shape[:2])

    plt.imshow(skel_img); plt.show(); plt.close()

    plt.figure(1)
    plt.clf()
    plt.subplot(231)
    plt.imshow(img)
    plt.title('input')
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(skel_img)
    plt.title('joint projection')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(rend_img_overlay)
    plt.title('3D Mesh overlay')
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(rend_img)
    plt.title('3D mesh')
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(rend_img_vp1)
    plt.title('diff vp')
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(rend_img_vp2)
    plt.title('diff vp')
    plt.axis('off')
    plt.draw()
    plt.show()
 
    #import ipdb
    #ipdb.set_trace()
    print("leaving function "+funcname)
    plt.close()


#=======================================================
def preprocess_image(img_path, json_path=None):
    img = io.imread(img_path)  # img is a numpy array.  Yusssss
    print("img.shape:\n{0}\n\n".format(img.shape))
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if json_path is None:
        if np.max(img.shape[:2]) != config.img_size:
            print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(json_path)

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img
#=======================================================
def preprocess_image_nathan(img, json_path=None):
    print("img.shape:\n{0}\n\n".format(img.shape))
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if json_path is None:
        if np.max(img.shape[:2]) != config.img_size:
            print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(json_path)

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img  # what the f**k was Kanazawa even using this 'img' variable at the end for?  Maybe it's just left over from old code.

#=======================================================
def main(img_path, json_path=None):
    sess = tf.Session()
    model = RunModel(config, sess=sess)

    input_img, proc_param, img = preprocess_image(img_path, json_path)  # resizing would happen HERE.

    # Add batch dimension: 1 x D x D x 3
    input_img = np.expand_dims(input_img, 0)

    # Theta is the 85D vector holding [camera, pose, shape]
    # where camera is 3D [s, tx, ty]
    # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
    # shape is 10D shape coefficients of SMPL
    joints, verts, cams, joints3d, theta = model.predict(
        input_img, get_theta=True)
    # REVISED NOTE: the first 69 params within the Theta (uppercase theta) are pose parameters.  There are actually 23 joints we're considering, and then each of the 23 joints somehow has only 3 parameters defining a rotation around that joint.  I'm guessing it's like roll, pitch, and yaw?  last 3 are (I THINK) the global 3x3 rotation matrix, a translation t (2x2 ie. (x,y)), 
    pn();pe()
    print("Saving parameters from this past run...")
    print("params are pose (size 69), shape (size 10), rotation(4), translation(2), and scale (1)")  # NO.  This would add up to 86, but our printed result says theta.shape==(1,85)
    # We employ the weak-perspective camera model and solve for the global rotation R (9) in (axis-angle representation (https://en.wikipedia.org/wiki/Axis-angle_representation)), translation t (2) and scale s (1).  Note: we STILL haven't figured out which part of the Theta is the shape parameters (Betas).  This is the next part we're looking for.    (nxb,   Wed Mar 13 12:00:53 EDT 2019)
    # roll, pitch and yaw should be possible (ie. compress rotation to 3 numbers instead of 4)  But I guess axis-angle representation is easier to compute??
    pe();pn();
    npy_fname=img_path[img_path.rfind('/')+1:]+"__Theta_params.npy"
    pr("numpy params Theta are:\n", theta)
    betas=theta[:,75:]
    pe();pe();pe();pr("shape params beta are\n {0}".format(betas));pe();pe();pe()


    pe();pn(); pr("Saving numpy params at "+npy_fname);pn();pe()
    np.save(img_path[img_path.rfind('/')+1:]+"__Theta_params.npy", theta)
    #np.save(img_path[img_path.rfind('/')+1:]+"__beta_params7080.npy" , theta[:,70:80])
    #np.save(img_path[img_path.rfind('/')+1:]+"__beta_params6979.npy" , theta[:,69:79])
    np.save(img_path[img_path.rfind('/')+1:]+"__beta_params.npy" , betas)
    # this just CAN'T be right.  It failed on something (looked totally different through render_SMPL.py than it did through this HMR)
    print('\n'*4+"saving vertices...")
    print("outmesh_path is ",outmesh_path); print('\n'*4)
    with open( outmesh_path, 'a') as fp:
      for vert_idx in range(verts.shape[1]):
        fp.write( 'v %f %f %f\n' % ( verts[0,vert_idx,0], verts[0,vert_idx,1], verts[0,vert_idx,2] ))
    fix()
    visualize(img, proc_param, joints[0], verts[0], cams[0])

    #print('verts.shape:\n',verts.shape) # (1, 6890, 3)
#====================== end main(params) (basically.  technically this isn't the end of main, but the rest ofthe crap below was just print statements and debugging) =====================

    d=dir;from pprint import pprint as p; #print(d(model)) #['E_var', '__class__', '__delattr__', '__dict__', '__doc__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'all_Js', 'all_cams', 'all_kps', 'all_verts', 'batch_size', 'build_test_model_ief', 'config', 'data_format', 'final_thetas', 'images_pl', 'img_feat', 'img_size', 'joint_type', 'load_path', 'mean_value', 'mean_var', 'model_type', 'num_cam', 'num_stage', 'num_theta', 'predict', 'predict_dict', 'prepare', 'proj_fn', 'saver', 'sess', 'smpl', 'smpl_model_path', 'total_params']
    #p(d(model.smpl)) # ['J_regressor', 'J_transformed', '__call__', '__class__', '__delattr__', '__dict__', '__doc__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'joint_regressor', 'num_betas', 'parents', 'posedirs', 'shapedirs', 'size', 'v_template', 'weights']
    #p(model.smpl.weights) # <tf.Variable 'lbs_weights:0' shape=(6890, 24) dtype=float32_ref>.  was the same regardless of which image I used
    #p(model.smpl.J_regressor) # <tf.Variable 'J_regressor:0' shape=(6890, 24) dtype=float32_ref>
    #p(model.smpl.num_betas) # 10 and 10
    ######################## NOTE NOTE NOTE NOTE ########################

    #print("model.smpl.beta: \n{0}".format(model.smpl.beta))  # no field .beta or .betas

    print("np.amax(verts) is {0}".format(np.amax(verts)))
    print("np.amin(verts) is {0}".format(np.amin(verts)))
    # NOTE:  it depends on the size of the human in the image!
    # for back.jpg:
    #  max is  0.694546461105
    #  min is -1.04786634445

    # for left.jpg:
    #  np.amax(verts) is  0.617345392704
    #  np.amin(verts) is -1.08882558346

    # for data/im1954.jpg, 
    #  np.amax(verts) is 0.487660825253
    #  np.amin(verts) is -1.08803486824  # NOTE: TINY!   next TODO: try with data/coco...
    # for data/im1963.jpg
    #  np.amax(verts) is 0.789033710957
    #  np.amin(verts) is -0.653137624264

    # NOTE: should be able to scale up/down the points by a simple multiplication.

    #print(verts.shape)  #(1,6890,3)   6890 vertices
    #verts[0,-1]=np.array([10000,-10,0])
    # ^ this line "verts[0,-1]=np.array([10000,-10,0])"  worked (as hoped) to fuck up a single vertex of the skin-model.
    # NOTE: this has the side effect of saving the faces at the end of the file name specified in outmesh_path
#====================== end main(params) =====================

#=============================================================
def make_mesh(img_path):
    # this was an old version for Pierlorenzo (~Feb 7-10, 2019)
    if os.path.isfile(outmesh_path):
      sp.call(['rm', outmesh_path]) # b/c old mesh

    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    main(img_path, config.json_path) # here, img_path is a parameter, not from config.
    # NOTE: oughta do this the right way, but right now the .obj file is getting written "backward."  So we just gotta rewrite it.  We do that in the fix() function
    with open(outmesh_path, 'r') as fp:
      return fp.read() # fp?  fp.read()?  NOTE: Pier said the actual file is preferable to just the string, but I'm not sure what he means by this.  What's the difference between the file and the string contents?
      #return fp.read()
#====================== end make_mesh(params) =====================




 

if __name__ == '__main__':
    if os.path.isfile(outmesh_path):
      sp.call(['rm', outmesh_path]) # b/c old mesh

    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    #main3(config.img_path, config.json_path)
    print("configs:             ",config)
    print("config.img_path:     ",config.img_path)
    print("config.json_path:    ",config.json_path)
    main(config.img_path, config.json_path)
    # NOTE: oughta do this the right way, but right now the .obj file is getting written "backward."  So we just gotta rewrite it.  We do that in the fix() function











































































































































