#! /usr/bin/env python

import argparse
# import ipdb as pdb
import cPickle as pickle

from loaders import load_imagenet_mean, load_rgb_hwc_mean_and_convert, load_labels, caffe
from jby_misc import WithTimer
from max_tracker import scan_images_for_maxes
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Finds images in a training set that cause max activation for a network; saves results in a pickled NetMaxTracker.')
    parser.add_argument('--N', type=int, default=9, help='note and save top N activations')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU id. Set -1 to use CPU')
    parser.add_argument('net_prototxt', type=str, default='', help='network prototxt to load')
    parser.add_argument('net_weights', type=str, default='', help='network weights to load')
    parser.add_argument('output_path', type=str, help='output filename for pkl')
    parser.add_argument('--mean-path', type=str, default=None, help='path to mean file')
    parser.add_argument('--images-dir', type=str, default='.', help='directory to look for files in. Used only together with --filelist-path')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--filelist-path', type=str, default=None,
                       help='list of image files to consider, one per line')
    group.add_argument('--images-csv-path', type=str, default=None,
                       help='path to csv file containing image pathes and labels')
    args = parser.parse_args()

    if args.mean_path is not None:
        print 'Loading RGB HWC mean and convert it to BRG CWH from', args.mean_path
        mean = load_rgb_hwc_mean_and_convert(args.mean_path)
    else:
        print 'Loading Imagenet mean'
        mean = load_imagenet_mean()

    if args.gpu_id >= 0:
        print 'Using GPU: {}'.format(args.gpu_id)
        caffe.set_device(args.gpu_id)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    net = caffe.Classifier(args.net_prototxt, args.net_weights,
                           mean=mean,
                           channel_swap=(2, 1, 0),
                           raw_scale=255,
                           image_dims=(256, 256))

    layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8_exh_all', 'prob']
    is_conv_layer = [('conv' in ll) for ll in layers]

    images_df = None
    if args.images_csv_path is not None:
        images_df = pd.read_csv(args.images_csv_path, index_col='index')

    with WithTimer('Scanning images'):
        max_tracker = scan_images_for_maxes(net, args.N,
                                            datadir=args.images_dir,
                                            filelist_path=args.filelist_path,
                                            images_df=images_df,
                                            layers=layers, is_conv_layer=is_conv_layer)
    with WithTimer('Saving maxes'):
        with open(args.output_path, 'wb') as ff:
            pickle.dump(max_tracker, ff, -1)



if __name__ == '__main__':
    main()
