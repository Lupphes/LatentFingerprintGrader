#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-

from ast import arg
import os
import argparse
import sys
import pathlib
import pickle

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

    return parser.parse_args()


def set_envinronment(args):
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.ext == "jp2":
        os.environ['OPENCV_IO_ENABLE_JASPER'] = "true"

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

    for filename in os.listdir(args.sdir):
        f = os.path.join(args.sdir, filename)
        if os.path.isfile(f) and pathlib.Path(f).suffix == '.' + args.ext:

            specific_folder = os.path.join(args.ddir, filename)
            if not os.path.exists(specific_folder):
                os.mkdir(specific_folder)

            fingerprint_image = fp.fingerprint.Fingerprint(path=f)
            pickle_path = os.path.join(specific_folder, filename + '.pickle')
            if args.regenerate or not os.path.exists(pickle_path):
                lf_latent = fingerprint_image.mus_afis_segmentation(
                    image_path=f, destination_dir=specific_folder, lf_latent=lf_latent)

                with open(pickle_path, 'wb') as handle:
                    pickle.dump(fingerprint_image, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
            else:
                print("woah")
                with open(pickle_path, 'rb') as handle:
                    fingerprint_image = pickle.load(handle)
            fingerprint_image.grade_fingerprint()
            fingerprint_image.generate_rating()


if __name__ == "__main__":
    args = argumentParse()
    set_envinronment(args)
    main(args)
