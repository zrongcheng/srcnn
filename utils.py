"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
import h5py
import random
import matplotlib.pyplot as plt

from PIL import Image  # for loading images as YCbCr format
import Scipy.misc
import Scipy.ndimage
import numpy as np

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS#全局变量
def read_data(path):
  """
  Read h5 format data file

  Args:
    path: file path of desired file所需文件
    data: '.h5' file format that contains train data values
    label: '.h5' file format that contains train label values
  """
  with h5py.File(path, 'r') as hf:
    data = np.array(hf.get('data'))
    label = np.array(hf.get('label'))
    return data, label

def preprocess(path, scale=3):
  """
  Preprocess single image file 
    (1) Read original image as YCbCr format (and grayscale as default)
    (2) Normalize
    (3) Apply image file with bicubic interpolation
    唯一的预处理操作

  Args:
    path: file path of desired file
    input_: image applied bicubic interpolation (low-resolution)
    label_: image with original resolution (high-resolution)
  """
  image = imread(path, is_grayscale=True)#？？为什么灰度处理
  label_ = modcrop(image, scale)

  # Must be normalized
  #image = image / 255.
  label_ = label_ / 255.
  #两次为了降低精度
  input_ = Scipy.ndimage.interpolation.zoom(label_, zoom=(1. / scale), prefilter=False)#一次
  input_ = Scipy.ndimage.interpolation.zoom(input_, zoom=(scale / 1.), prefilter=False)#二次，bicubic
  #imsave(input_,r'F:\tf_py\srcnn\sample\test1.png')
  #imsave(label_, r'F:\tf_py\srcnn\sample\test2.png')
  return input_, label_

def prepare_data(dataset):
  """
  Args:
    dataset: choose train dataset or test dataset
    
    For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
  """
  if FLAGS.is_train:
    #filenames = os.listdir(dataset)
    data_dir = os.path.join(os.getcwd(), dataset)
    # glob.glob()获取当前目录或相对路径所有文件的路径，输出一个list，读取字符中的*(通配符)
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
  else:
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)), "Set5")
    data = glob.glob(os.path.join(data_dir, "*.bmp"))

  return data

def make_data(data, label):
  """
  Make input data as h5 file format
  Depending on 'is_train' (flag value), savepath would be changed.
  """
  if FLAGS.is_train:
    savepath = os.path.join(os.getcwd(), 'checkpoint/train.h5')
  else:
    savepath = os.path.join(os.getcwd(), 'checkpoint/test.h5')

  with h5py.File(savepath, 'w') as hf:
    hf.create_dataset('data', data=data)
    hf.create_dataset('label', data=label)

def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    return Scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)#将图像转灰度
  else:
    return Scipy.misc.imread(path, mode='YCbCr').astype(np.float)#默认为false

def modcrop(image, scale=3):
  """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  
  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  要缩放原始图像, 首先要做的是在缩放操作时没有余数。

  我们需要找到高度 (和宽度) 和比例因子的模。
  然后, 减去原始图像大小的高度 (和宽度) 的模。
  即使在缩放操作之后, 也不会有余数。
  """
  if len(image.shape) == 3:#彩色 800*600*3
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:#灰度 800*600
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

def input_setup(config):
  """
  Read image files and make their sub-images and saved them as a h5 file format.
  """
  # Load data path
  if config.is_train:
    data = prepare_data(dataset="Train")
  else:
    data = prepare_data(dataset="Test")

  sub_input_sequence = []
  sub_label_sequence = []
  padding = abs(config.image_size - config.label_size) // 2 # 6 填充

  if config.is_train:
    for i in range(len(data)):
      input_, label_ = preprocess(data[i], config.scale)#data[i]为数据目录

      if len(input_.shape) == 3:
        h, w, _ = input_.shape
      else:
        h, w = input_.shape
      #。。。
      for x in range(0, h-config.image_size+1, config.stride):
        for y in range(0, w-config.image_size+1, config.stride):
          sub_input = input_[x:x+config.image_size, y:y+config.image_size] # [33 x 33]
          sub_label = label_[x+padding:x+padding+config.label_size,
                      y+padding:y+padding+config.label_size] # [21 x 21]

          # Make channel value,颜色通道1
          sub_input = sub_input.reshape([config.image_size, config.image_size, 1])
          sub_label = sub_label.reshape([config.label_size, config.label_size, 1])

          sub_input_sequence.append(sub_input)
          sub_label_sequence.append(sub_label)
    # Make list to numpy array. With this transform
    arrdata = np.asarray(sub_input_sequence)  # [?, 33, 33, 1]
    arrlabel = np.asarray(sub_label_sequence)  # [?, 21, 21, 1]

    make_data(arrdata, arrlabel)  # 把处理好的数据进行存储，路径为checkpoint/..
  else:
    input_, label_ = preprocess(data[4], config.scale)

    if len(input_.shape) == 3:
      h, w, _ = input_.shape
    else:
      h, w = input_.shape
    input = input_.reshape([h,w,1])

    label=label_[6:h-6,6:w-6]
    label=label.reshape([h-12,w-12,1])

    sub_input_sequence.append(input)
    sub_label_sequence.append(label)

    input1 = np.asarray(sub_input_sequence)
    label1 = np.asarray(sub_label_sequence)
    #label=label_.reshape([height,weight,1])
    return input1,label1,h,w
    # # Numbers of sub-images in height and width of image are needed to compute merge operation.
    # nx = ny = 0
    # for x in range(0, h-config.image_size+1, config.stride):
    #   nx += 1; ny = 0
    #   for y in range(0, w-config.image_size+1, config.stride):
    #     ny += 1
    #     sub_input = input_[x:x+config.image_size, y:y+config.image_size] # [33 x 33]
    #     sub_label = label_[x+padding:x+padding+config.label_size,
    #                 y+padding:y+padding+config.label_size] # [21 x 21]
    #
    #     sub_input = sub_input.reshape([config.image_size, config.image_size, 1])
    #     sub_label = sub_label.reshape([config.label_size, config.label_size, 1])
    #
    #     sub_input_sequence.append(sub_input)
    #     sub_label_sequence.append(sub_label)

  """
  len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
  (sub_input_sequence[0]).shape : (33, 33, 1)
  """



  # if not config.is_train:
  #   return nx, ny
    
def imsave(image, path):
  return Scipy.misc.imsave(path, image)

# def merge(images, size):
#   h, w = images.shape[1], images.shape[2]#21*21
#   p,q,j=0,0,0
#   img = np.zeros((14*(size[0]-1)+21, 14*(size[1]-1)+21, 1))
#   for idx, image in enumerate(images):#image.shape=(21,21,1)
#     i = idx % size[1]#余数
#     t=j
#     j = idx // size[1]#商
#     if (j-t)==1:
#       p=p+14
#       q=0
#     #img[0:21,0:21,:]=image
#     img[p:p+h, q:q+w, :] = image
#
#     q=q+14
#
#   return img
