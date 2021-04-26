#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 2021/4/26-14:35
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : superior_god
# @File     : dcgan_face_generation.py
# @Project  : 00PythonProjects

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from imageio import imsave, mimsave

from models import discriminator, generator, sigmoid_cross_entropy_with_logits
from utils import read_image, montage

parser = argparse.ArgumentParser()
parser.add_argument('--auther', help='inner batch size', default="TuringEmmy", type=str)
parser.add_argument('--IS_DRAW', help='weather to show graph', action='store_true', default=False)
parser.add_argument('--OUTPUT', help='weather to show graph', default=".", type=str)
parser.add_argument('--BATCH_SIZE', help='weather to show graph', default=100, type=int)
parser.add_argument('--Z_DIM', help='weather to show graph', default=100, type=int)
parser.add_argument('--WIDTH', help='weather to show graph', default=64, type=int)
parser.add_argument('--HEIGHT', help='weather to show graph', default=64, type=int)
parser.add_argument('--EPOCHS', help='weather to show graph', default=60000, type=int)
parser.add_argument('--LEARNING_RATE', help='weather to show graph', default=0.0002, type=float)
parser.add_argument('--PER_PLOT', help='weather to show graph', default=500, type=int)
parser.add_argument('--DATASETS_NAME', help='lfw_new_imgs  or celeba datasets', default="lfw_new_imgs", type=str)

args = parser.parse_args()

dataset = args.OUTPUT + "/lfw_new_imgs"
images = glob.glob(os.path.join(dataset, '*.*'))
print(len(images))

OUTPUT_DIR = args.OUTPUT + '/samples_' + args.DATASETS_NAME
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

X = tf.placeholder(dtype=tf.float32, shape=[None, args.HEIGHT, args.WIDTH, 3], name='X')
noise = tf.placeholder(dtype=tf.float32, shape=[None, args.Z_DIM], name='noise')
is_training = tf.placeholder(dtype=tf.bool, name='is_training')

g = generator(noise)
d_real, d_real_logits = discriminator(X)
d_fake, d_fake_logits = discriminator(g, reuse=True)

vars_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
vars_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]

loss_d_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_real_logits, tf.ones_like(d_real)))
loss_d_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_fake_logits, tf.zeros_like(d_fake)))
loss_g = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_fake_logits, tf.ones_like(d_fake)))
loss_d = loss_d_real + loss_d_fake

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer_d = tf.train.AdamOptimizer(learning_rate=args.LEARNING_RATE, beta1=0.5).minimize(loss_d, var_list=vars_d)
    optimizer_g = tf.train.AdamOptimizer(learning_rate=args.LEARNING_RATE, beta1=0.5).minimize(loss_g, var_list=vars_g)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    z_samples = np.random.uniform(-1.0, 1.0, [args.BATCH_SIZE, args.Z_DIM]).astype(np.float32)
    samples = []
    loss = {'d': [], 'g': []}

    offset = 0
    for i in range(args.EPOCHS):
        n = np.random.uniform(-1.0, 1.0, [args.BATCH_SIZE, args.Z_DIM]).astype(np.float32)

        offset = (offset + args.BATCH_SIZE) % len(images)
        batch = np.array([read_image(img, args.HEIGHT, args.WIDTH) for img in images[offset: offset + args.BATCH_SIZE]])
        batch = (batch - 0.5) * 2

        d_ls, g_ls = sess.run([loss_d, loss_g], feed_dict={X: batch, noise: n, is_training: True})
        loss['d'].append(d_ls)
        loss['g'].append(g_ls)

        sess.run(optimizer_d, feed_dict={X: batch, noise: n, is_training: True})
        sess.run(optimizer_g, feed_dict={X: batch, noise: n, is_training: True})
        sess.run(optimizer_g, feed_dict={X: batch, noise: n, is_training: True})

        if i % args.PER_PLOT == 0:
            print(i, d_ls, g_ls)
            gen_imgs = sess.run(g, feed_dict={noise: z_samples, is_training: False})
            gen_imgs = (gen_imgs + 1) / 2
            imgs = [img[:, :, :] for img in gen_imgs]
            gen_imgs = montage(imgs)
            plt.axis('off')
            if args.IS_DRAW:
                plt.imshow(gen_imgs)
            imsave(os.path.join(OUTPUT_DIR, 'sample_%d.jpg' % i), gen_imgs)
            if args.IS_DRAW:
                plt.show()
            samples.append(gen_imgs)

    plt.plot(loss['d'], label='Discriminator')
    plt.plot(loss['g'], label='Generator')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(OUTPUT_DIR, 'Loss.png'))
    if args.IS_DRAW:
        plt.show()
    mimsave(os.path.join(OUTPUT_DIR, 'samples.gif'), samples, fps=10)

    saver = tf.train.Saver()
    saver.save(sess, os.path.join(OUTPUT_DIR, 'dcgan_' + args.DATASETS_NAME), global_step=60000)
