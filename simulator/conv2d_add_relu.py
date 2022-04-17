
import numpy as np
from util import *
from mem import *
import tensorflow as tf
from conv2d_kernel import *
import math
import pdb

def test(params):
    util = Util()
    mem = MEM()
    # N,Hi,Wi,Ci,R,S,Co,Ho,Wo = 128,56,56,64,3,3,64,56,56
    # strides = [1,1]
    # padding = [1,1,1,1]
    # dilation = [1,1]
    N,Hi,Wi,Ci,R,S,Co,Ho,Wo,strides,padding,dilation = params

    # init
    np.random.seed(0)
    # tf.disable_v2_behavior()
    f = tf.placeholder(tf.float32, shape=[N,Hi,Wi,Ci])
    f_val = np.random.randint(10, size=(N,Hi,Wi,Ci))
    w = tf.placeholder(tf.float32, shape=[R,S,Ci,Co])
    w_val = np.random.randint(10, size=(R,S,Ci,Co))
    bias_val = np.random.randint(10, size=(1,1,1,Co))
    # standard implementation
    cpu_result = tf.nn.conv2d(f, w, strides=[1, 1, 1, 1], padding='SAME')
    
    feed_dict_f32 = {f: f_val, w: w_val}
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        cpu_result_val = sess.run([cpu_result], feed_dict_f32)
    # print(cpu_result_val)
    print(np.array(cpu_result_val).shape)
    # import pdb;pdb.set_trace()
    res_std = cpu_result_val[0] + bias_val
    res_std = np.maximum(np.zeros_like(res_std), res_std)

    # # conv_loop
    # conv2d = Conv2dKernel()
    # res = mem.new([N,Ho,Wo,Co], "zero")
    # res = conv2d.conv_loop(params, f_val, w_val, bias_val, res)
    # util.check_result(res_std, res, " conv_loop ")

    # step0
    conv2d = Conv2dKernel()
    res = mem.new([N,Ho,Wo,Co], "zero")
    res = conv2d.step0(params, f_val, w_val, bias_val, res)
    util.check_result(res_std, res, " step0 ")
    return

#'''
if __name__ == '__main__':
    # N,Hi,Wi,Ci,R,S,Co,Ho,Wo = 128,56,56,64,3,3,64,56,56
    N,Hi,Wi,Ci,R,S,Co,Ho,Wo = 16,56,56,64,3,3,64,56,56
    strides = [1,1]
    padding = [1,1,1,1]
    dilation = [1,1]
    params = N,Hi,Wi,Ci,R,S,Co,Ho,Wo,strides,padding,dilation
    test(params)