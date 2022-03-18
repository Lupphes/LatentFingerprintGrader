#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-

from ast import arg
import os
import argparse
import sys
import pathlib
import pickle
import glob
import matplotlib.pylab as plt

import fingerprint_tools as fp
from fingerprint_tools.exception import ArrgumentError as ArrgumentError


def argumentParse():
    """ Parse the arguments """
    parser = argparse.ArgumentParser(
        add_help=True,
        prog="demo",
        description='Demo for ..., Author: Ondřej Sloup (xsloup02)'
    )

    parser.add_argument(
        '-g', '--gpu', help='comma separated list of GPU(s) to use.', default='0'
    )

    parser.add_argument(
        '-e', '--ext', type=str, help='extenstion of selected file', default='jp2'
    )

    parser.add_argument(
        '-s', '--sdir', type=pathlib.Path,
        help='Path to location where extracted templates should be stored', required=True
    )
    parser.add_argument(
        '-d', '--ddir', type=pathlib.Path, help='Path to directory containing input images'
    )
    parser.add_argument(
        '-r', '--regenerate', help='Regenerate the latent scan', action='store_true'
    )
    parser.add_argument(
        '-p', '--dpi', type=int, help='DPI of the images', default=500
    )

    return parser.parse_args()


def set_envinronment(args):
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.ext == "jp2":
        os.environ['OPENCV_IO_ENABLE_JASPER'] = "true"

    # Setting environment vars for tensorflow
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Switching matplotlib backend
    plt.switch_backend('agg')

    if not os.path.exists(args.sdir):
        raise ArrgumentError("Specified path doesn't exist")

    if args.ddir == None:
        i = 0
        while os.path.exists(os.path.basename(args.sdir) + "_out_" + str(i)):
            i += 1
        os.mkdir(os.path.join(os.path.dirname(args.sdir), os.path.basename(
            args.sdir) + "_out_" + str(i)))
        args.ddir = pathlib.Path(
            os.path.basename(args.sdir) + "_out_" + str(i))

    if not os.path.exists(args.ddir):
        os.mkdir(args.ddir)


def main(args):
    """ Launcher for Fingerprint tool package """

    lf_latent = None

    inputpath = args.sdir
    outputpath = args.ddir

    for dirpath, dirnames, filenames in os.walk(inputpath):
        for file in filenames:
            structure = os.path.join(
                outputpath, os.path.relpath(dirpath, inputpath))
            sub_dir = os.path.join(structure, file)
            if pathlib.Path(sub_dir).suffix == '.' + args.ext:
                if not os.path.isdir(structure):
                    os.makedirs(structure)
                source_image = os.path.join(dirpath, file)
                image_dir = os.path.join(os.path.dirname(sub_dir), file)
                if not os.path.isdir(image_dir):
                    os.mkdir(image_dir)

                fingerprint_image = fp.fingerprint.Fingerprint(
                    path=source_image, dpi=args.dpi)
                pickle_path = os.path.join(image_dir, file + '.pickle')
                if args.regenerate or not os.path.exists(pickle_path):
                    lf_latent = fingerprint_image.mus_afis_segmentation(
                        image_path=source_image, destination_dir=image_dir, lf_latent=lf_latent)

                    with open(pickle_path, 'wb') as handle:
                        pickle.dump(fingerprint_image, handle,
                                    protocol=pickle.HIGHEST_PROTOCOL)
                    pass
                else:
                    print("woah")
                    with open(pickle_path, 'rb') as handle:
                        fingerprint_image = pickle.load(handle)
                fingerprint_image.grade_fingerprint()
                fingerprint_image.generate_rating(os.path.dirname(image_dir))
                fingerprint_image.generate_images(image_dir, ".jpeg")


if __name__ == "__main__":
    args = argumentParse()
    set_envinronment(args)
    main(args)
