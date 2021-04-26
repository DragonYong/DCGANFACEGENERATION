#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 2021/4/26-14:33
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : superior_god
# @File     : server.py
# @Project  : 00PythonProjects
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from utils import montage

parser = argparse.ArgumentParser()
parser.add_argument('--auther', help='inner batch size', default="TuringEmmy", type=str)
parser.add_argument('--OUTPUT', help='weather to show graph', default=".", type=str, required=True)
parser.add_argument('--BATCH_SIZE', help='weather to show graph', default=128, type=int)
parser.add_argument('--Z_DIM', help='weather to show graph', default=100, type=int)
parser.add_argument('--DATASETS_NAME', help='lfw_new_imgs  or celeba datasets', default="lfw_new_imgs", type=str)

args = parser.parse_args()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    model_path = args.OUTPUT + '/samples_' + args.DATASETS_NAME
    saver = tf.train.import_meta_graph(
        os.path.join(model_path, 'dcgan_' + args.DATASETS_NAME + '-60000.meta'))
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    graph = tf.get_default_graph()
    g = graph.get_tensor_by_name('generator/g/Tanh:0')
    noise = graph.get_tensor_by_name('noise:0')
    is_training = graph.get_tensor_by_name('is_training:0')

    n = np.random.uniform(-1.0, 1.0, [args.BATCH_SIZE, args.Z_DIM]).astype(np.float32)
    gen_imgs = sess.run(g, feed_dict={noise: n, is_training: False})
    gen_imgs = (gen_imgs + 1) / 2
    imgs = [img[:, :, :] for img in gen_imgs]
    gen_imgs = montage(imgs)
    gen_imgs = np.clip(gen_imgs, 0, 1)
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(gen_imgs)
    plt.show()
