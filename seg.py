
# TODO: rename everything "seg" rather than "segment"
WHITE=255
BLACK=0
TRANSPARENT=0

import cv2
import os, tarfile, tempfile
from glob import glob
import numpy as np
np.seterr(all='raise')
import tensorflow as tf
import scipy
from scipy.ndimage.measurements import center_of_mass as CoM
from io import BytesIO
import imageio as ii
import skimage
from PIL import Image

import sys
import subprocess as sp

from d import debug
from save import save
from viz import pltshow
from utils import np_img, round_tuple, neg, no_color_shift, pn

if debug:
  import matplotlib
  matplotlib.use('Agg')      
  from matplotlib import pyplot as plt  # NOTE:  problem on Nathan's machine (Dec. 21, 2018) is JUST with pyplot.  None of the rest of matplotlib is a problem AT ALL.
from matplotlib import gridspec
from six.moves import urllib
from scipy.ndimage import shift  # TODO: make all imports precisely like THIS.  from math import pi.  It's short, it's searchable (debuggable), 

from copy import deepcopy

'''
import matplotlib
matplotlib.use('Agg')      
'''
# NOTE: The above lines didn't end up being necessary.    But it's a good FYI so future programmers can solve matplotlib display problems, though

#================================================================
class DeepLabModel(object):
  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    self.graph = tf.Graph()
    graph_def = None
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')
    self.sess = tf.Session(graph=self.graph)
#================================================================
  def run(self, image):
    #print(type(image)) # type(image) is    <class 'PIL.PngImagePlugin.PngImageFile'>
    width, height = image.size
    resize_ratio = float(self.INPUT_SIZE / max(width, height))
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map
#================================================================
  def segment_nparr(self, img):
    '''
      Segments many shapes of images
      In the method, we resize to a shape (ie. (513,288)) where one of the dimensions is 513 is required
    '''
    # tODO: understand better
    width, height, RGB = img.shape
    resize_ratio = float(self.INPUT_SIZE / max(width, height)) # 513/ bigger of width and height
    target_size = (int(resize_ratio * width), int(resize_ratio * height), 3)
    resized_image = skimage.transform.resize(img,target_size, anti_aliasing=True)

    # return segmentation mask
    return (self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [resized_image]})[0], 
      resized_image)
  #=============== end segment_nparr(self, img): ===============
  #================================================================
# end class definition DeepLabModel()
#================================================================


#================================================================
def create_pascal_label_colormap():
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap
#================================================================
def label_to_color_image(label):
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')
  colormap = create_pascal_label_colormap()
  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')
  return colormap[label]


#================================================================
def binarize(mask_3_colors):
  RED=0; CERTAIN=256.0; probable = np.array(int(CERTAIN / 2)-1) # default color is magenta, but the red part shows 
  mask_binary = deepcopy(mask_3_colors[:,:,RED])
  return mask_binary.astype('bool')
# end binarize(mask_3_colors):
#================================================================
def run_visualization(url, model):
  try:
    f = urllib.request.urlopen(url)
    jpeg_str = f.read()
    original_im = Image.open(BytesIO(jpeg_str))
  except IOError:
    print('Cannot retrieve image. Please check url: ' + url)
    return
  if debug:
    print('running deeplab on image %s...' % url)
  resized_im, seg_map = model.run(original_im)
  if save:
    ii.imwrite("_segmented____binary_mask_.jpg", seg_map)  
    # I think something in the saving process ***ks up the mask with noise.  Dec. 14, 2018
  #PROBABLE = 127  # NOTE:  experimental from 1 data point;  PLEASE DOUBLE CHECK if u get a noisy segmentation
  #ii.imwrite("_segmented____binary_mask_.jpg", np.greater(seg_map, PROBABLE).astype('bool'))
  return np.rot90(seg_map,k=3) # k=3 b/c it gives us the result we want   (I tested it experimentally.  Dec. 26, 2018)   # this is true for the URL version, not the other
#===== end run_visualization(url): =====
#================================================================
def seg_map(img, model):
  print('running deeplab on image')
  seg_map, resized_im = model.segment_nparr(img) # within def seg_map(img, model)
  if debug:
    pltshow(seg_map)
  if save:
    fname = "_segmented____binary_mask_.jpg"
    print('saving segmentation map in ', fname)
    ii.imwrite(fname, seg_map)  # Dec. 14, 2018:  I think something in the saving process ***ks up the mask with noise
  return np.rot90(seg_map,k=3) # k=3 b/c it gives us the result we want   (I tested it experimentally.  Dec. 26, 2018)
#=========== end ============ seg_map(img, model): ==============  # NOTE: only need to do this if func is REALLY long.

#================================================================
def seg(img):
  #================================================================
  LABEL_NAMES = np.asarray([
      'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
      'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
      'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
  ])
  #================================================================
  FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
  FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
  MODEL_NAME = 'mobilenetv2_coco_voctrainaug'
  _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
  _MODEL_URLS = {
      'mobilenetv2_coco_voctrainaug': 'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
      'mobilenetv2_coco_voctrainval': 'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz'
  }
  _TARBALL_NAME = 'deeplab_model.tar.gz'
  model_dir = './'
  download_path = os.path.join(model_dir, _TARBALL_NAME)
  # urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME], download_path)
  MODEL = DeepLabModel(download_path) # segment_local()
  FAIRLY_CERTAIN=127
  return seg_map(img, MODEL)
#=====  end seg(params) =====
#================================================================
def segment_local(local_filename):
  img=np.asarray(ii.imread(local_filename))
  if debug:
    pltshow(img)
  img=img.astype("float64")
  #================================================================
  #================================================================
  LABEL_NAMES = np.asarray([
      'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
      'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
      'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
  ])
  #================================================================
  #================================================================
  FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
  FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
  MODEL_NAME = 'mobilenetv2_coco_voctrainaug'
  _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
  _MODEL_URLS = {
      'mobilenetv2_coco_voctrainaug': 'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
      'mobilenetv2_coco_voctrainval': 'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz'
  }
  _TARBALL_NAME = 'deeplab_model.tar.gz'
  model_dir = './'
  download_path = os.path.join(model_dir, _TARBALL_NAME)
  # urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME], download_path)
  MODEL = DeepLabModel(download_path) # segment_local()
  FAIRLY_CERTAIN=127
  return seg_map(img, MODEL)
#=====  end segment_local(local_filename) =====
#seg_local = segment_local   # instead of using this "seg_local = segment_local," I did "from seg import segment_local as seg_local"

#================================================================
#==================================================
def segment_URL(IMG_URL):
  '''
    NOTE: segmentation requires internet connection
  '''
  # TODO:   allow us to set the URL from parameter
  LABEL_NAMES = np.asarray([
      'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
      'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
      'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
  ])

  FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
  FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
  MODEL_NAME = 'mobilenetv2_coco_voctrainaug'

  _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
  _MODEL_URLS = {
      'mobilenetv2_coco_voctrainaug': 'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
      'mobilenetv2_coco_voctrainval': 'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz'
  }
  _TARBALL_NAME = 'deeplab_model.tar.gz'

  model_dir = './'
  download_path = os.path.join(model_dir, _TARBALL_NAME)
  # urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME], download_path)

  MODEL = DeepLabModel(download_path)

  FAIRLY_CERTAIN=127
  return np.greater(run_visualization(IMG_URL, MODEL), FAIRLY_CERTAIN)
#===== end segment_URL(IMG_URL): =====




#====================================================================
def segment_black_background(local_fname):
  # NOTE: WORKING.
  '''
    BLACK is 0; so this function can be used to "add" two images together to superimpose them.
  '''
  # TODO: debug, generalize, etc.  Something ain't working (one of the last parts)
  # NOTE:  PIL.resize(shape)'s shape has the width and height in the opposite order from numpy's height and width
  # NOTE: weird sh!t happened when I tried to convert to 'float64' in the `np.array(Image.open(fname))` line.
  segmap  = segment_local(local_fname) # segment_black_background()
  segmap  = segmap.reshape(segmap.shape[0],segmap.shape[1],1)
  segmap  = np.rot90(       np.concatenate((segmap,segmap,segmap),axis=2)        )
  # I really OUGHT to scale the mask to fit the dimensions of the image (we'd have better resolution this way)
  img     = Image.open(local_fname)
  img     = img.resize((segmap.shape[1],segmap.shape[0]), Image.ANTIALIAS)
  img     = np.array(img) 
  if debug:
    pltshow(img); pltshow(segmap)
  # as of (Wed Feb 20 17:49:37 EST 2019), segmap is 0 for background, 15 for human ( before astype('bool'))
  segmap=np.logical_not(segmap.astype('bool'))

  # cut out the human from the img
  img[segmap]=BLACK
  if debug:
    pltshow(img)
  fname='person_cutout__black_background.png'
  ii.imwrite(fname,img)
  return img, np.logical_not(segmap)
  # logical_not() because mask should be where there's a person, not where there's background
#========== end segment_black_background(params): ==========

#========================================================================
if __name__=='__main__':
  print('\n'*2)
  if len(sys.argv) == 1:
    IMG_URL = 'http://columbia.edu/~nxb2101/180.0.png'
    #'http://vishalanand.net/green.jpg'
    print ("\nusage: python2 seg.py [url_of_img_containing_human(s)] \n  example: python2 seg.py http://vishalanand.net/green.jpg   \n\n")
    print ("currently segmenting image found at url: \n  "+IMG_URL)
    pltshow(segment_URL(IMG_URL))
  elif len(sys.argv) == 2:
    img_path=sys.argv[1]
    #IMG_URL = sys.argv[1]# TODO: uncomment to segment images on the internet.
    #segmap=segment_local(sys.argv[1])
    pltshow(np_img(sys.argv[1]))
    #pltshow(np.rot90(np_img(sys.argv[1]),k=1))
    no_background, segmap = segment_black_background(sys.argv[1])
    pltshow(no_background)
    pltshow(segmap.astype('float64'))
  else:
    fnames=sys.argv[1:] # TODO: change the analogous code in render_smpl.py
    # TODO: if we extend seg.py (processing more cmd line args and more configuration parameters) the line `fnames=[sys.argv[1:]]` will break.
    #   A better way to do it is with `import absl; absl.configs`
    #     (see akanazawa's HMR for example of these configs.)

    # TODO: error checking:
    #   if len(sys.argv) < 3:   # should it be less than 3?  more?
    overlay_imgs(fnames[0], fnames[1])  # tODO: hardcode the fnames?
#========================== end __main__ =====================================




















































  '''
  #segmap = segment_URL(IMG_URL) # TODO: uncomment to segment images on the internet.
  segment_transparent_background(img_path)
  print('now doing black')
  segment_black_background(img_path)
  # TODO: more thorough tests.  For now, I'm going to move on b/c I wanna sleep and there are more important things.  But if you think of any bugs, please fix ASAP rather than 3 months from now.
  '''





















# Glossary:
'''
  Function definitions (function headers)

  As of Sat Mar  2 07:39:17 EST 2019,

    47:  def __init__(self, tarball_path):
    63:  def run(self, image):
    75:  def segment_nparr(self, img):
    98:def create_pascal_label_colormap():
    109:def label_to_color_image(label):
    119:def binarize(mask_3_colors):
    125:def run_visualization(url, model):
    144:def seg_map(img, model):
    157:def segment_local(local_filename):
    188:def segment_URL(IMG_URL):
    224:def segment_black_background(local_fname):

  tags glossary gloss defs funcs britney bitch
'''


