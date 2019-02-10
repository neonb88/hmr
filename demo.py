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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np
import subprocess as sp
import os

import skimage.io as io
import tensorflow as tf

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel

flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')

outmesh_path = '/home/ubuntu/x/p/fresh____as_of_Dec_12_2018/vr_mall____fresh___Dec_12_2018/src/web/upload_img_flask/mesh.obj'

def fix():
  # flips .obj file so faces are after vertices
  fresh_outmesh_path  = 'freshmesh.obj'
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
def visualize(img, proc_param, joints, verts, cam):
    """
    Renders the result in original image coordinate frame.
    """
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

    import matplotlib.pyplot as plt
    # plt.ion()
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
    # import ipdb
    # ipdb.set_trace()


def preprocess_image(img_path, json_path=None):
    img = io.imread(img_path)
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


def main(img_path, json_path=None):
    sess = tf.Session()
    model = RunModel(config, sess=sess)

    input_img, proc_param, img = preprocess_image(img_path, json_path)  # resizing would happen HERE.
    import viz
    viz.pltshow(input_img)
    viz.pltshow(img)

    # Add batch dimension: 1 x D x D x 3
    input_img = np.expand_dims(input_img, 0)

    # Theta is the 85D vector holding [camera, pose, shape]
    # where camera is 3D [s, tx, ty]
    # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
    # shape is 10D shape coefficients of SMPL
    joints, verts, cams, joints3d, theta = model.predict(
        input_img, get_theta=True)
    # NOTE: SIDE EFFECT.  REFACTOR!
    with open( outmesh_path, 'a') as fp:
      print('\n'*4+"saving vertices...")
      print("outmesh_path is ",outmesh_path); print('\n'*4)
      for vert_idx in range(verts.shape[1]):
        fp.write( 'v %f %f %f\n' % ( verts[0,vert_idx,0], verts[0,vert_idx,1], verts[0,vert_idx,2] ))

    """
    print('verts.shape:\n',verts.shape) # (1, 6890, 3)
    print('cams:\n',cams)
    print('joints:\n',joints)
    print('joints3d:\n',joints3d)
    print('theta:\n',theta)
    """

    d=dir;from pprint import pprint as p; #print(d(model)) #['E_var', '__class__', '__delattr__', '__dict__', '__doc__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'all_Js', 'all_cams', 'all_kps', 'all_verts', 'batch_size', 'build_test_model_ief', 'config', 'data_format', 'final_thetas', 'images_pl', 'img_feat', 'img_size', 'joint_type', 'load_path', 'mean_value', 'mean_var', 'model_type', 'num_cam', 'num_stage', 'num_theta', 'predict', 'predict_dict', 'prepare', 'proj_fn', 'saver', 'sess', 'smpl', 'smpl_model_path', 'total_params']
    #p(d(model.smpl)) # ['J_regressor', 'J_transformed', '__call__', '__class__', '__delattr__', '__dict__', '__doc__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'joint_regressor', 'num_betas', 'parents', 'posedirs', 'shapedirs', 'size', 'v_template', 'weights']
    #p(model.smpl.weights) # <tf.Variable 'lbs_weights:0' shape=(6890, 24) dtype=float32_ref>.  was the same regardless of which image I used
    #p(model.smpl.J_regressor) # <tf.Variable 'J_regressor:0' shape=(6890, 24) dtype=float32_ref>
    #p(model.smpl.num_betas) # 10 and 10

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
    visualize(img, proc_param, joints[0], verts[0], cams[0])
    # NOTE: this has the side effect of saving the faces at the end of the file name specified in outmesh_path


if __name__ == '__main__':
    if os.path.isfile(outmesh_path):
      sp.call(['rm', outmesh_path])

    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    main(config.img_path, config.json_path)
    # NOTE: oughta do this the right way, but right now the .obj file is getting written "backward."  So we just gotta rewrite it.  We do that in the fix() function
    fix()











































































































































