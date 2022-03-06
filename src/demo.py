#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-

import os
import argparse
import sys
import pathlib

import fingerprint_tools as fp


def argumentParse():
    return 0


def main(args):
    """ Launcher for Fingerprint tool package """

    image_latent = 'B111.jp2'
    image_exemplar = '002-06.jp2'

    name = image_latent

    ###############

    path = "img/" + name
    fingerprint_image = fp.fingerprint.Fingerprint(path=path, name=name)
    # fingerprint_image.mus_afis_segmentation()
    fingerprint_image.filter_fingerprint()


if __name__ == "__main__":
    args = argumentParse()
    main(args)
