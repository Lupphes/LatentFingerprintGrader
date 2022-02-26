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

    image_path = 'img/012_3_1.png'
    fingerprint_image = fp.fingerprint.Fingerprint(path=image_path)
    fingerprint_image.grade_fingerprint()


if __name__ == "__main__":
    args = argumentParse()
    main(args)
