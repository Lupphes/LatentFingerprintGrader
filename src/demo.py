#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-

import os
import argparse
import sys
import pathlib

import fingerprint_tools as fp


def argumentParse():
    pass


def main(args):
    """ Launcher for Fingerprint tool package """
    fp.fingerprint.Fingerprint.grade_fingerprints()


if __name__ == "__main__":
    args = argumentParse()
    main(args)
