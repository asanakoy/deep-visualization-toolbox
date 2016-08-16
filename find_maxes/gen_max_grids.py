#! /usr/bin/env python
# Artsiom Sanakoyeu, Aug 2016

import argparse
import os
import scipy.misc
import sys
import numpy as np

def get_unit_ids(layer_dir):
    """
    Find ids of all units in layer
    :param layer_dir - dir for layer with dirs per unit
    """
    import re
    import glob
    pattern_unit_dir = os.path.join(layer_dir, 'unit_*')
    dir_pathes = glob.glob(pattern_unit_dir)
    expr_model = r'unit_(\d+)\Z'
    ids = []
    for path in dir_pathes:
        match_obj = re.search(expr_model, path)
        assert match_obj is not None
        ids.append(int(match_obj.group(1)))
    ids.sort()
    return ids


def stitch_images(image_pathes, padding=2):
    """
    Stitch images in a square grid adding lines of $padding px between them
    :param image_pathes:
    :param padding:
    :return:
    """
    import math
    n = len(image_pathes)
    n_root = int(math.floor(math.sqrt(n)))
    if n_root * n_root != n:
        raise NotImplementedError('Cannot create a grid from {} images'.format(n))

    images = []
    for path in image_pathes:
        images.append(scipy.misc.imread(path))
    img_side = images[0].shape[0]
    for img in images:
        assert img.shape == (img_side, img_side, 3), \
            'Bad image shape {} != {}'.format(img.shape, (img_side, img_side, 3))

    grid_width = n_root * img_side + (n_root - 1) * padding
    rows = []
    for i in xrange(n_root):
        row_list = []
        for j in xrange(n_root):
            row_list.append(images[i * n_root + j])
            if j != n_root - 1:
                row_list.append(np.zeros((img_side, padding, 3), dtype=int))
        rows.append(np.hstack(row_list))
        if i != n_root - 1:
            rows.append(np.zeros((padding, grid_width, 3)))
    grid_image = np.vstack(rows)
    return grid_image


def main():
    parser = argparse.ArgumentParser(
        description='Loads a pickled NetMaxTracker and outputs one or more of {the patches of the image, a deconv patch, a backprop patch} associated with the maxes.')
    parser.add_argument('--N', type=int, default=9, help='Note and save top N activations.')
    parser.add_argument('--do-maxim', action='store_true', help='Output max patches.')
    parser.add_argument('--do-deconv', action='store_true', help='Output deconv patches.')

    parser.add_argument('crops_dir', type=str, help='Directory to look for crops per layer per unit')
    parser.add_argument('out_dir', type=str,help='out dir')
    parser.add_argument('--layer', type=str, help='Which layer to output')
    args = parser.parse_args()

    layer_dir = os.path.join(args.crops_dir, args.layer)
    unit_ids = get_unit_ids(layer_dir)

    crop_path_tpml = os.path.join(layer_dir, 'unit_{:04d}/{}_{:03d}.png')

    group_names = []
    crop_prefixes = []
    if args.do_maxim:
        group_names.append('max_im')
        crop_prefixes.append('maxim')
    if args.do_deconv:
        group_names.append('max_deconv')
        crop_prefixes.append('deconv')
    assert len(group_names) > 0

    # For ex: out_dir/max_im/conv2/conv2_0009.jpg
    out_path_tmpl = os.path.join(args.out_dir, '{{}}/{0}/{0}_{{:04d}}.jpg'.format(args.layer))

    for group, prefix in zip(group_names, crop_prefixes):
        print 'Group: {}'.format(group)
        for i, unit_id in enumerate(unit_ids):
            sys.stdout.write('\rUnit {:04d}/{}'.format(i, len(unit_ids)))
            sys.stdout.flush()
            crop_pathes_list = [crop_path_tpml.format(unit_id, prefix, i) for i in xrange(args.N)]
            grid_img = stitch_images(crop_pathes_list)
            out_img_path = out_path_tmpl.format(group, unit_id)
            if not os.path.exists(os.path.dirname(out_img_path)):
                os.makedirs(os.path.dirname(out_img_path))
            scipy.misc.imsave(out_img_path, grid_img)

        print '---'


if __name__ == '__main__':
    main()
