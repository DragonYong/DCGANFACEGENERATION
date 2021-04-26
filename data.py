#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 2021/4/26-15:52
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : superior_god
# @File     : data.py
# @Project  : 00PythonProjects
import argparse
import os
import tarfile
import urllib

from imageio import imread, imsave

parser = argparse.ArgumentParser()
parser.add_argument('--auther', help='inner batch size', default="TuringEmmy", type=str)
parser.add_argument('--OUTPUT_DIR', help='weather to show graph', default=".")
parser.add_argument('--DATA', help='weather to show graph', default=".")

args = parser.parse_args()


def untgz_file(url, filename, directory, new_dir):
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)

    if not os.path.isdir(directory):
        if not os.path.isfile(filename):
            urllib.request.urlretrieve(url, filename)
        tar = tarfile.open(filename, 'r:gz')
        tar.extractall(path=directory)
        tar.close()

    count = 0
    for dir_, _, files in os.walk(directory):
        for file_ in files:
            img = imread(os.path.join(dir_, file_))
            imsave(os.path.join(new_dir, '%d.png' % count), img)
            count += 1


if __name__ == '__main__':
    # 下载和处理LFW数据
    url = 'http://vis-www.cs.umass.edu/lfw/lfw.tgz'
    filename = args.DATA + '/lfw.tgz'
    directory = args.OUTPUT_DIR + '/lfw_imgs'
    new_dir = args.OUTPUT_DIR + '/lfw_new_imgs'
    untgz_file(url, filename, directory, new_dir)
