""" Evaluates a trained model using placeholders. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from os.path import exists
import os

# nxb's imports:
import sys
from pprint import pprint as p
import matplotlib.pyplot as plt
import pickle as pkl

from .tf_smpl import projection as proj_util
from .tf_smpl.batch_smpl import SMPL
from .models import get_encoder_fn_separate

class RunModel(object):
    def __init__(self, config, sess=None):
        """
        Args:
          config
          The session is initialized to the base session.
        """
        self.config = config
        self.load_path = config.load_path
        
        # Config + path
        if not config.load_path:
            raise Exception(
                "[!] You need to specify `load_path` to load a pretrained model"
            )
        if not exists(config.load_path + '.index'):
            print('%s doesnt exist..' % config.load_path)
            import ipdb
            ipdb.set_trace()

        # Data
        self.batch_size = config.batch_size
        self.img_size = config.img_size

        self.data_format = config.data_format
        self.smpl_model_path = config.smpl_model_path
 
        input_size = (self.batch_size, self.img_size, self.img_size, 3)
        self.images_pl = tf.placeholder(tf.float32, shape=input_size)

        # Model Settings
        self.num_stage = config.num_stage
        self.model_type = config.model_type
        self.joint_type = config.joint_type
        # Camera
        self.num_cam = 3
        self.proj_fn = proj_util.batch_orth_proj_idrot

        self.num_theta = 72        
        # Theta size: camera (3) + pose (24*3) + shape (10)
        self.total_params = self.num_cam + self.num_theta + 10

        self.smpl = SMPL(self.smpl_model_path, joint_type=self.joint_type)  
        # NOTE:  call to actual SMPL() object is here (SMPL(...params)).  Can we get the betas out?
        #Reply: as of Tue Mar  5 07:48:17 EST 2019, I think the betas inside are initially tensorflow variables, not numpy (ie. their values are not set in stone until AFTER this funccall concludes.)
        """
        print(self.smpl.__class__)
        print('class of obj from hello_smpl.py is <class \'chumpy.ch_ops.add\'>')
        print('\n'*3)
        """

        # self.theta0_pl = tf.placeholder_with_default(
        #     self.load_mean_param(), shape=[self.batch_size, self.total_params], name='theta0')
        # self.theta0_pl = tf.placeholder(tf.float32, shape=[None, self.total_params], name='theta0')

        self.build_test_model_ief()

        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess
        if not os.path.exists('summaries'):
            os.mkdir('summaries')
        if not os.path.exists(os.path.join('summaries','first')):
            os.mkdir(os.path.join('summaries','first'))
        
        # Load data.
        self.saver = tf.train.Saver()
        self.prepare()        


    def build_test_model_ief(self):
        # Load mean value
        self.mean_var = tf.Variable(tf.zeros((1, self.total_params)), name="mean_param", dtype=tf.float32)

        img_enc_fn, threed_enc_fn = get_encoder_fn_separate(self.model_type)
        # Extract image features.        
        self.img_feat, self.E_var = img_enc_fn(self.images_pl,
                                               is_training=False,
                                               reuse=False)
        
        # Start loop
        self.all_verts = []
        self.all_kps = []
        self.all_cams = []
        self.all_Js = []
        self.final_thetas = []
        theta_prev = tf.tile(self.mean_var, [self.batch_size, 1])
        for i in np.arange(self.num_stage):
            print('Iteration %d' % i)
            # ---- Compute outputs
            state = tf.concat([self.img_feat, theta_prev], 1)

            if i == 0:
                delta_theta, _ = threed_enc_fn(
                    state,
                    num_output=self.total_params,
                    is_training=False,
                    reuse=False)
            else:
                delta_theta, _ = threed_enc_fn(
                    state,
                    num_output=self.total_params,
                    is_training=False,
                    reuse=True)

            # Compute new theta
            theta_here = theta_prev + delta_theta
            # cam = N x 3, pose N x self.num_theta, shape: N x 10
            cams = theta_here[:, :self.num_cam]                
            poses = theta_here[:, self.num_cam:(self.num_cam + self.num_theta)]
            shapes = theta_here[:, (self.num_cam + self.num_theta):]

            verts, Js, _ = self.smpl(shapes, poses, get_skin=True)

            # Project to 2D!
            pred_kp = self.proj_fn(Js, cams, name='proj_2d_stage%d' % i)
            self.all_verts.append(verts)
            self.all_kps.append(pred_kp)
            self.all_cams.append(cams)
            self.all_Js.append(Js)
            # save each theta.
            self.final_thetas.append(theta_here)
            # Finally)update to end iteration.
            theta_prev = theta_here


    def prepare(self):
        print('Restoring checkpoint %s..' % self.load_path)
        self.saver.restore(self.sess, self.load_path)        
        self.mean_value = self.sess.run(self.mean_var)
            
    def predict(self, images, get_theta=False):
        # NOTE: VHMR can be tried by extending this function.
        # is [this] (see below) a shape of a numpy array?  Is type(images)=='np.array'?
        #images: num_batch, img_size, img_size, 3
        """
        images: num_batch, img_size, img_size, 3
        Preprocessed to range [-1, 1]
        """
        results = self.predict_dict(images)
        if get_theta:
            return results['joints'], results['verts'], results['cams'], results[
                'joints3d'], results['theta']
        else:
            return results['joints'], results['verts'], results['cams'], results[
                'joints3d']

    #======================================================================================
        #======================================================================================
        #========================== IMPORTANT FUNCTION predict_dict() =========================
        #======================================================================================
        #                 it gets the joint locations.
    def predict_dict(self, images):
        """
        images: num_batch, img_size, img_size, 3
        Preprocessed to range [-1, 1]
        Runs the model with images.
        """
        plt.imshow(images.reshape(224,224,3));  plt.show();  plt.close()
        print("images.shape is {0}".format(images.shape)) # images.shape is (1, 224, 224, 3)

        feed_dict = {
            self.images_pl: images,
            # self.theta0_pl: self.mean_var,
        }
        fetch_dict = {
            'joints': self.all_kps[-1],  # maybe just don't fetch this "joints" variable, and instead turn absl.config.json_path into the old format HMR requests.
            'verts': self.all_verts[-1],
            'cams': self.all_cams[-1],
            'joints3d': self.all_Js[-1],
            'theta': self.final_thetas[-1],
        }
        print("fetch_dict is :")
        p(fetch_dict)

        '''
        with open('tf_sess.pkl', 'wb') as f:     
          pkl.dump(self.sess, f)

        # I think what's happening is it's impossible to pickle a class inside of a different module   than the name of that class.
        '''
        summ_writer = tf.summary.FileWriter(os.path.join('summaries','first'), self.sess.graph)

        results = self.sess.run(fetch_dict, feed_dict)  # This "(self.sess.run(fetch_dict, feed_dict))" must be called before we can get the betas out of smpl?
        # the real question is: CAN we even get the betas out?  predict_dict() predicts the values of joints, verts, cams, joints3d, and theta just fine, but can we relearn the betas from these values?
        """
        'joints': self.all_kps[-1],
        'verts': self.all_verts[-1],
        'cams': self.all_cams[-1],
        'joints3d': self.all_Js[-1],
        'theta': self.final_thetas[-1],
        """

        # Return joints in original image space.        
          # (ie. we ran openpose on the 224x224 img, then resized them to the original image's size.)   
            # -nxb, Fri Mar 29 11:21:27 EDT 2019
        joints = results['joints']
        joints=((joints + 1) * 0.5) * self.img_size
        print("joints.shape:",joints.shape) # (1,19,2)
        plt.scatter(joints[:,:,0], joints[:,:,1]); plt.show(); plt.close()
        results['joints'] = joints

        return results
    #======================== end func predict_dict(params) ===============================










































































































































