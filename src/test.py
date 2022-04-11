#!/usr/bin/env python3.7
# coding: utf-8

import argparse
from pathlib import Path
import json
from enum import Enum
import numpy as np
from matplotlib import pyplot as plt
from typing import Dict


def argument_parse() -> argparse.ArgumentParser:
    """
    Parse the arguments for the whole script and
    pass them to main
    """
    parser = argparse.ArgumentParser(
        add_help=True,
        prog='LatFigGra Test',
        description='LatFigGra (Latent Fingerprint Grader) Test Script 2022, Author: Ondřej Sloup (xsloup02)'
    )
    parser.add_argument(
        '-s', '--sdir', type=Path,
        help='Path to the input log.json script', required=True
    )
    parser.add_argument(
        '-d', '--ddir', type=Path, help='Path to the destination folder, where the script will store images and logs.', default='test/'
    )
    return parser.parse_args()


class Classification(str, Enum):
    GOOD = 'G',
    BAD = 'B',
    UGLY = 'U',
    UNKNOWN = 'UNK'


def report2dict(rating, report, quatity, rmse, n_rig, sin_s, thick, sin_s_core, thick_core):
    if quatity != None:
        report[rating]['min_points'].append(quatity)
    if rmse != None:
        report[rating]['rmse'].append(rmse)
    if n_rig != None:
        report[rating]['n_rig'].append(n_rig)
    if sin_s != None:
        report[rating]['sin_s'].append(sin_s)
    if thick != None:
        report[rating]['thick'].append(thick)
    if sin_s_core != None:
        report[rating]['sin_s_core'].append(sin_s_core)
    if thick_core != None:
        report[rating]['thick_core'].append(thick_core)


def generateJSON(report, output_path, filename='log_test.json'):
    with open(output_path / filename, 'w') as file:
        json.dump(report, file, indent=4)


def save_fig(dictionary: Dict, path: Path, ext: str) -> None:
    for variant in dictionary:
        fname = path / f"{variant}{ext}"
        dictionary[variant].savefig(fname)


def main(args: argparse.ArgumentParser) -> None:
    log_file_path: Path = args.sdir
    output_path: Path = args.ddir

    report = {
        Classification.GOOD: {
            'min_points': [],
            'rmse': [],
            'n_rig': [],
            'sin_s': [],
            'thick': [],
            'sin_s_core': [],
            'thick_core': []
        },
        Classification.BAD: {
            'min_points': [],
            'rmse': [],
            'n_rig': [],
            'sin_s': [],
            'thick': [],
            'sin_s_core': [],
            'thick_core': []
        },
        Classification.UGLY: {
            'min_points': [],
            'rmse': [],
            'n_rig': [],
            'sin_s': [],
            'thick': [],
            'sin_s_core': [],
            'thick_core': []
        },
        Classification.UNKNOWN: {
            'min_points': [],
            'rmse': [],
            'n_rig': [],
            'sin_s': [],
            'thick': [],
            'sin_s_core': [],
            'thick_core': []
        },
    }

    with open(log_file_path, 'r+') as file:
        file_data = json.load(file)
        number_of_prints = 0
        for print_name in file_data:
            number_of_prints += 1
            quality_class = print_name[:1]

            quatity = file_data[print_name]['minutuae_points']['quantity']
            if not 'contrast' in file_data[print_name]:
                rmse = None
            else:
                rmse = file_data[print_name]['contrast']['rmse']

            if not 'papilary_ridges' in file_data[print_name]:
                n_rig = None
            else:
                n_rig = file_data[print_name]['papilary_ridges']['total_mean']

            if not 'papillary_crosscut' in file_data[print_name]:
                sin_s = None
                thick = None
            else:
                dict_sin = file_data[print_name]['papillary_crosscut']['sinusoidal_shape']
                dict_thick = file_data[print_name]['papillary_crosscut']['thickness']
                if not 'aec' in dict_sin or len(dict_sin['aec']['D_D_ridges']) == 0:
                    sin_s = None
                else:
                    sin_s = np.mean(dict_sin['aec']['D_D_ridges'])

                if not 'aec_core' in dict_sin or len(dict_sin['aec_core']['D_D_ridges']) == 0:
                    sin_s_core = None
                else:
                    sin_s_core = np.mean(dict_sin['aec_core']['D_D_ridges'])

                if not 'aec' in dict_thick or len(dict_thick['aec']['thickness_difference']) == 0:
                    thick = None
                else:
                    thick = np.mean(dict_thick['aec']['thickness_difference'])

                if not 'aec_core' in dict_thick or len(dict_thick['aec_core']['thickness_difference']) == 0:
                    thick_core = None
                else:
                    thick_core = np.mean(
                        dict_thick['aec_core']['thickness_difference'])

            report2dict(quality_class, report, quatity, rmse,
                        n_rig, sin_s, thick, sin_s_core, thick_core)

    generateJSON(report=report, output_path=output_path)

    color_legend = {'Good': u'green', 'Bad': u'orange',
                    'Ugly': u'red', 'Unknown': u'black'}
    figures = {}

    # --------------------------------------------------------------------
    minutiae_point: plt.Figure = plt.figure(figsize=(22, 5), dpi=150)

    total_displayed = 0
    rating_array = []

    for quality_class in report:
        for rating in report[quality_class]['min_points']:
            color = 'black'
            if quality_class == Classification.GOOD:
                color = 'green'
            elif quality_class == Classification.BAD:
                color = 'orange'
            elif quality_class == Classification.UGLY:
                color = 'red'

            plt.plot(total_displayed, rating, marker="o",
                     color=color, figure=minutiae_point)

            total_displayed += 1

        rating_array += report[quality_class]['n_rig']
    rating_len = len(rating_array)

    z = np.polyfit(np.arange(rating_len), rating_array, 1)
    p = np.poly1d(z)
    plt.plot(np.arange(rating_len), p(np.arange(rating_len)), "r--")

    plt.title('Minutia points')
    plt.xlabel('Fingeprint number', fontsize='small', figure=minutiae_point)
    plt.ylabel('Number of minutiae', fontsize='small', figure=minutiae_point)
    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='')
               for color in color_legend.values()]
    plt.legend(markers, color_legend.keys(), numpoints=1, fontsize='small')
    plt.rc('font', size=14)
    plt.close()
    figures['minutiae_points_rating'] = minutiae_point

    # --------------------------------------------------------------------
    rmse: plt.Figure = plt.figure(figsize=(22, 5), dpi=150)

    total_displayed = 0
    rating_array = []

    for quality_class in report:
        for rating in report[quality_class]['rmse']:
            color = 'black'
            if quality_class == Classification.GOOD:
                color = 'green'
            elif quality_class == Classification.BAD:
                color = 'orange'
            elif quality_class == Classification.UGLY:
                color = 'red'

            plt.plot(total_displayed, rating, marker="o",
                     color=color, figure=rmse)

            total_displayed += 1

        rating_array += report[quality_class]['n_rig']

    rating_len = len(rating_array)

    z = np.polyfit(np.arange(rating_len), rating_array, 1)
    p = np.poly1d(z)
    plt.plot(np.arange(rating_len), p(np.arange(rating_len)), "r--")

    plt.title('RMSE')
    plt.xlabel('Fingeprint number', fontsize='small', figure=rmse)
    plt.ylabel('RMSE', fontsize='small', figure=rmse)
    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='')
               for color in color_legend.values()]
    plt.legend(markers, color_legend.keys(), numpoints=1, fontsize='small')
    plt.rc('font', size=14)
    plt.close()
    figures['rmse_rating'] = rmse

    # --------------------------------------------------------------------
    n_rig: plt.Figure = plt.figure(figsize=(22, 5), dpi=150)

    total_displayed = 0
    rating_array = []

    for quality_class in report:
        for rating in report[quality_class]['n_rig']:
            color = 'black'
            if quality_class == Classification.GOOD:
                color = 'green'
            elif quality_class == Classification.BAD:
                color = 'orange'
            elif quality_class == Classification.UGLY:
                color = 'red'

            plt.plot(total_displayed, rating, marker="o",
                     color=color, figure=n_rig)

            total_displayed += 1

        rating_array += report[quality_class]['n_rig']

    rating_len = len(rating_array)

    z = np.polyfit(np.arange(rating_len), rating_array, 1)
    p = np.poly1d(z)
    plt.plot(np.arange(rating_len), p(np.arange(rating_len)), "r--")

    plt.title('Number of ridges')
    plt.xlabel('Fingeprint number', fontsize='small', figure=n_rig)
    plt.ylabel('Number of ridges', fontsize='small', figure=n_rig)
    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='')
               for color in color_legend.values()]
    plt.legend(markers, color_legend.keys(), numpoints=1, fontsize='small')
    plt.rc('font', size=14)
    plt.close()
    figures['number_of_ridges_rating'] = n_rig

    # --------------------------------------------------------------------

    sin_s: plt.Figure = plt.figure(figsize=(22, 5), dpi=150)

    total_displayed = 0

    for quality_class in report:
        for rating in report[quality_class]['sin_s']:
            color = 'black'
            if quality_class == Classification.GOOD:
                color = 'green'
            elif quality_class == Classification.BAD:
                color = 'orange'
            elif quality_class == Classification.UGLY:
                color = 'red'

            plt.plot(total_displayed, rating, marker="o",
                     color=color, figure=sin_s)

            total_displayed += 1

        rating_array = report[quality_class]['sin_s']
        rating_len = len(rating_array)

        if rating_len != 0:
            plt.plot(np.arange(rating_len) + total_displayed - rating_len, np.full(rating_len,
                     np.mean(rating_array)), color='black', linestyle='--', label='Mean value')

    plt.title('Sinusoidal crosscut aec')
    plt.xlabel('Fingeprint number', fontsize='small', figure=sin_s)
    plt.ylabel('Sinusoidal deviance', fontsize='small', figure=sin_s)
    plt.axhline(y=0, color='r', linestyle='-')
    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='')
               for color in color_legend.values()]
    plt.legend(markers, color_legend.keys(), numpoints=1, fontsize='small')
    plt.rc('font', size=14)
    plt.close()

    figures['sinusoidal_similarity_rating'] = sin_s

    # --------------------------------------------------------------------

    thick: plt.Figure = plt.figure(figsize=(22, 5), dpi=150)

    total_displayed = 0

    for quality_class in report:
        for rating in report[quality_class]['thick']:
            color = 'black'
            if quality_class == Classification.GOOD:
                color = 'green'
            elif quality_class == Classification.BAD:
                color = 'orange'
            elif quality_class == Classification.UGLY:
                color = 'red'

            plt.plot(total_displayed, rating, marker="o",
                     color=color, figure=thick)

            total_displayed += 1

        rating_array = report[quality_class]['sin_s']
        rating_len = len(rating_array)

        if rating_len != 0:
            plt.plot(np.arange(rating_len) + total_displayed - rating_len, np.full(rating_len,
                     np.mean(rating_array)), color='black', linestyle='--', label='Mean value')

    plt.title('Thickness aec')
    plt.xlabel('Fingeprint number', fontsize='small', figure=thick)
    plt.ylabel('Sinusoidal deviance', fontsize='small', figure=thick)
    plt.axhline(y=0, color='r', linestyle='-')
    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='')
               for color in color_legend.values()]
    plt.legend(markers, color_legend.keys(), numpoints=1, fontsize='small')
    plt.rc('font', size=14)
    plt.close()

    figures['thickness_rating'] = thick

    # --------------------------------------------------------------------

    sin_s_core: plt.Figure = plt.figure(figsize=(22, 5), dpi=150)

    total_displayed = 0

    for quality_class in report:
        for rating in report[quality_class]['sin_s_core']:
            color = 'black'
            if quality_class == Classification.GOOD:
                color = 'green'
            elif quality_class == Classification.BAD:
                color = 'orange'
            elif quality_class == Classification.UGLY:
                color = 'red'

            plt.plot(total_displayed, rating, marker="o",
                     color=color, figure=sin_s_core)

            total_displayed += 1

        rating_array = report[quality_class]['sin_s']
        rating_len = len(rating_array)

        if rating_len != 0:
            plt.plot(np.arange(rating_len) + total_displayed - rating_len, np.full(rating_len,
                     np.mean(rating_array)), color='black', linestyle='--', label='Mean value')

    plt.title('Sinusoidal crosscut CORE aec')
    plt.xlabel('Fingeprint number', fontsize='small', figure=sin_s_core)
    plt.ylabel('Sinusoidal deviance', fontsize='small', figure=sin_s_core)
    plt.axhline(y=0, color='r', linestyle='-')
    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='')
               for color in color_legend.values()]
    plt.legend(markers, color_legend.keys(), numpoints=1, fontsize='small')
    plt.rc('font', size=14)
    plt.close()

    figures['sinusoidal_similarity_core_rating'] = sin_s_core

    # --------------------------------------------------------------------

    thick_core: plt.Figure = plt.figure(figsize=(22, 5), dpi=150)

    total_displayed = 0

    for quality_class in report:
        for rating in report[quality_class]['thick_core']:
            color = 'black'
            if quality_class == Classification.GOOD:
                color = 'green'
            elif quality_class == Classification.BAD:
                color = 'orange'
            elif quality_class == Classification.UGLY:
                color = 'red'

            plt.plot(total_displayed, rating, marker="o",
                     color=color, figure=thick_core)

            total_displayed += 1

        rating_array = report[quality_class]['sin_s']
        rating_len = len(rating_array)

        if rating_len != 0:
            plt.plot(np.arange(rating_len) + total_displayed - rating_len, np.full(rating_len,
                     np.mean(rating_array)), color='black', linestyle='--', label='Mean value')

    plt.title('Thickness CORE aec')
    plt.xlabel('Fingeprint number', fontsize='small', figure=thick_core)
    plt.ylabel('Sinusoidal deviance', fontsize='small', figure=thick_core)
    plt.axhline(y=0, color='r', linestyle='-')
    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='')
               for color in color_legend.values()]
    plt.legend(markers, color_legend.keys(), numpoints=1, fontsize='small')
    plt.rc('font', size=14)
    plt.close()

    figures['thickness_core_rating'] = thick_core

    save_fig(figures, output_path, '.jpeg')


if __name__ == "__main__":
    args = argument_parse()
    main(args)
