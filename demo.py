#!/usr/bin/env python3.7
# coding: utf-8

import os
import argparse
from pathlib import Path
import pickle
import logging
from datetime import datetime

import latfiggra as lfg
from latfiggra.exception import ArrgumentError as ArrgumentError


def argument_parse() -> argparse.ArgumentParser:
    """
    Parse the arguments for the whole script and
    pass them to the main
    """

    parser = argparse.ArgumentParser(
        add_help=True,
        prog='LatFigGra',
        description='LatFigGra (Latent Fingerprint Grader) 2022, Author: Ondřej Sloup (xsloup02)'
    )
    parser.add_argument(
        '-g', '--gpu', type=str, help='Comma-separated list of graphic cards which the script will use for msu_afis. By default, `0`.', default='0'
    )
    parser.add_argument(
        '-e', '--ext', type=str, help='File extension to which format the script will generate the output images. By default, `jp2`.', default='jp2'
    )
    parser.add_argument(
        '-s', '--sdir', type=Path,
        help='Path to the input folder, where the source images should be.', required=True
    )
    parser.add_argument(
        '-d', '--ddir', type=Path, help='Path to the destination folder, where the script will store fingerprint images and logs.'
    )
    parser.add_argument(
        '-m', '--models', type=Path, help='Path to the model folder. By default, `./models`.', default='models'
    )
    parser.add_argument(
        '-r', '--regenerate', help='Flag to regenerate already computed fingerprints (their pickle files) despite their existence.', action='store_true'
    )
    parser.add_argument(
        '-p', '--ppi', type=int, help='PPI (Pixels per inch) under which the scanner scanned the fingerprints. By default, 500.', default=500
    )

    return parser.parse_args()


def set_envinronment(args) -> None:
    """
    Prepares the python environment for packages and sets essential
    variables for the MSU_AFIS packages, and prepares arguments
    and directory structure
    """
    # Set up logging
    logging.basicConfig(
        filename=f"LatFigGra_{datetime.utcnow().strftime('%m.%d.%Y_%H.%M.%S')}.log",
        # filename=None,
        level=logging.INFO,
        format='%(levelname)s:[%(asctime)s] - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] - %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')
    logger.setFormatter(formatter)
    logging.getLogger('').addHandler(logger)
    logger = logging.getLogger(__name__)

    # Sets GPU for neural network
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # JASPER (JP2000) has vulnerabilities and needs
    # to be explicitely turned on
    # https://github.com/opencv/opencv/issues/14058
    if args.ext == 'jp2':
        os.environ['OPENCV_IO_ENABLE_JASPER'] = 'true'
        logging.warning(
            'Jasper project has many opened vulnerabilities. Beware what are you opening!'
        )

    # Setting environment vars for tensorflow
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Switch to matplotlib agg backend
    from matplotlib import pyplot as plt
    plt.switch_backend('agg')

    if not args.sdir.exists():
        raise FileNotFoundError(
            'Folder specified as source doesn\'t exist. Please check your source directory.'
        )

    # Creates output folder if not specified
    if args.ddir == None:
        i: int = 0
        args.ddir = (args.sdir).with_name(args.sdir.name + f'_out_{i}')
        while args.ddir.exists():
            i += 1
            args.ddir = (args.sdir).with_name(args.sdir.name + f'_out_{i}')


def main(args: argparse.ArgumentParser) -> None:
    """ 
    Main function of LatFigGra (Latent Fingerprint Grader) 
    Ondřej Sloup (xsloup02)
    Algorithmic Evaluation of the Quality of Dactyloscopic Traces
    Bachelor's Thesis 2022
    🦊 🐺 🦌 🐕

    Launch the script and recursively go through the given folders
    and generate log.json with fingerprint evaluation

    """

    # Variable for MSU_AFIS package
    # Used to load into memory just once
    msu_afis = None

    path_input: Path = args.sdir
    path_output: Path = args.ddir

    for dirpath, _, filenames in os.walk(path_input):
        for file in filenames:
            structure: str = os.path.join(
                path_output, os.path.relpath(dirpath, path_input))
            sub_dir: str = os.path.join(structure, file)

            if Path(sub_dir).suffix == '.' + args.ext:

                path_image_src: str = os.path.join(dirpath, file)
                path_img_des_dir: str = os.path.join(
                    os.path.dirname(sub_dir), file)
                path_pickle: str = os.path.join(
                    path_img_des_dir, file + '.pickle')

                if not os.path.isdir(path_img_des_dir):
                    os.makedirs(path_img_des_dir)

                # Create a fingerprint object
                fingerprint_image = lfg.fingerprint.Fingerprint(
                    path=path_image_src, ppi=args.ppi
                )

                # Restore or generate pickle file for faster calculation
                if args.regenerate or not os.path.exists(path_pickle):
                    msu_afis = fingerprint_image.msu_afis(
                        path_image=path_image_src, path_destination=path_img_des_dir, path_config=args.models, extractor_class=msu_afis, ext='.jpeg')  # args.ext

                    with open(path_pickle, 'wb') as handle:
                        pickle.dump(fingerprint_image, handle,
                                    protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    with open(path_pickle, 'rb') as handle:
                        fingerprint_image = pickle.load(handle)

                # Grade fingerprint, rating and images
                fingerprint_image.grade_fingerprint()
                fingerprint_image.generate_rating(
                    os.path.dirname(path_img_des_dir))
                fingerprint_image.generate_images(
                    path_img_des_dir, '.jpeg')  # args.ext


def mathplotlib_settings():
    import matplotlib
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })


if __name__ == "__main__":
    args = argument_parse()
    set_envinronment(args)
    mathplotlib_settings()
    main(args)
