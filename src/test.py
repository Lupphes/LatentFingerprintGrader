#!/usr/bin/env python3.7
# coding: utf-8

import argparse
from pathlib import Path
from enum import IntEnum
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import stats

from typing import Dict


class Classification(IntEnum):
    """
    Classification by the SD27 dataset
    """
    GOOD = 0,
    BAD = 1,
    UGLY = 2,
    UNKNOWN = 3,


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


def json2pandas(log_file_path: Path) -> pd.DataFrame:
    """
    Transforms the data from JSON to pandas and makes it usable 
    for analysis
    """

    df = pd.read_json(log_file_path, orient='index')
    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'image'})

    df['minutiae_points'] = pd.json_normalize(
        df['minutiae_points'])['quantity']
    df['papilary_ridges'] = pd.json_normalize(
        df['papilary_ridges'])['total_mean']
    papillary_crosscut = pd.json_normalize(
        df['papillary_crosscut']).dropna()
    contrast = pd.json_normalize(df['contrast'])

    df['rmse'] = contrast['rmse']
    df['michelson'] = contrast['michelson_contrast_pct']

    df['sin_s'] = papillary_crosscut['sinusoidal_shape.aec.D_D_ridges']
    df['sin_s'] = df['sin_s'][df['sin_s'].map(bool)].dropna()
    df['sin_s'] = [np.mean(x) for x in df['sin_s'].to_numpy()]

    df['sin_s_core'] = papillary_crosscut['sinusoidal_shape.aec_core.D_D_ridges']
    df['sin_s_core'] = df['sin_s_core'][df['sin_s_core'].map(bool)].dropna()
    df['sin_s_core'] = [np.mean(x) for x in df['sin_s_core'].to_numpy()]

    df['sin_s_gray'] = papillary_crosscut['sinusoidal_shape.gray.D_D_ridges']
    df['sin_s_gray'] = df['sin_s_gray'][df['sin_s_gray'].map(bool)].dropna()
    df['sin_s_gray'] = [np.mean(x) for x in df['sin_s_gray'].to_numpy()]

    df['sin_s_core_gray'] = papillary_crosscut['sinusoidal_shape.gray_core.D_D_ridges']
    df['sin_s_core_gray'] = df['sin_s_core_gray'][df['sin_s_core_gray'].map(
        bool)].dropna()
    df['sin_s_core_gray'] = [np.mean(x)
                             for x in df['sin_s_core_gray'].to_numpy()]

    df['thick'] = papillary_crosscut['thickness.aec.thickness_difference']
    df['thick'] = df['thick'][df['thick'].map(bool)].dropna()
    df['thick'] = [np.mean(x)for x in df['thick'].to_numpy()]

    df['thick_core'] = papillary_crosscut['thickness.aec_core.thickness_difference']
    df['thick_core'] = df['thick_core'][df['thick_core'].map(bool)].dropna()
    df['thick_core'] = [np.mean(x) for x in df['thick_core'].to_numpy()]

    df['thick_gray'] = papillary_crosscut['thickness.gray.thickness_difference']
    df['thick_gray'] = df['thick_gray'][df['thick_gray'].map(bool)].dropna()
    df['thick_gray'] = [np.mean(x) for x in df['thick_gray'].to_numpy()]

    df['thick_core_gray'] = papillary_crosscut['thickness.gray_core.thickness_difference']
    df['thick_core_gray'] = df['thick_core_gray'][df['thick_core_gray'].map(
        bool)].dropna()
    df['thick_core_gray'] = [np.mean(x)
                             for x in df['thick_core_gray'].to_numpy()]

    df = df.drop(labels='papillary_crosscut', axis=1)
    df = df.drop(labels='contrast', axis=1)
    df = df.drop(labels='error', axis=1)

    image_class = []
    for print_name in df['image']:
        quality_class = Classification.UNKNOWN
        letter = print_name[:1]
        if letter == 'G':
            quality_class = Classification.GOOD
        elif letter == 'B':
            quality_class = Classification.BAD
        elif letter == 'U':
            quality_class = Classification.UGLY
        image_class.append(quality_class)

    df['image_class'] = image_class

    conditions = [
        (df['image_class'] == Classification.GOOD),
        (df['image_class'] == Classification.BAD),
        (df['image_class'] == Classification.UGLY),
        (df['image_class'] == Classification.UNKNOWN)
    ]

    df['image_class']

    values = ['green', 'orange', 'red', 'black']
    df['color'] = np.select(conditions, values)

    df = df.sort_values('image_class')
    df.reset_index(inplace=True)

    return df


def plot_trendline(rating_array: pd.DataFrame, figure: plt.Figure) -> None:
    """
    Plot trendline based on the least squares polynomial fit 
    of the first degree
    """
    rating_array = rating_array.dropna()
    rating_len = len(rating_array)
    z = np.polyfit(np.arange(rating_len), rating_array, 1)
    p = np.poly1d(z)
    plt.plot(np.arange(rating_len), p(np.arange(rating_len)),
             "r--", figure=figure, label="Trendline")


def plot_metadata(title, x_label: str, y_label: str, figure: plt.Figure) -> None:
    """
    Data for the plot to generate nice charts
    """

    color_legend = {'Good': u'green', 'Bad': u'orange',
                    'Ugly': u'red'}  # , 'Unknown': u'black'

    plt.title(title)
    plt.xlabel(x_label, fontsize='small', figure=figure)
    plt.ylabel(y_label, fontsize='small', figure=figure)
    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='')
               for color in color_legend.values()]
    plt.legend(markers, color_legend.keys(), numpoints=1, fontsize='small')
    plt.rc('font', size=14)
    plt.close()


def plot_zero_line(str_dev: pd.DataFrame, value: str, figure: plt.Figure) -> None:
    """
    Plots 3 lines based on the asserted quality by the SD27 dataset
    They are calculated with their mean value
    """

    good_df: pd.DataFrame = str_dev.loc[str_dev['image_class']
                                        == Classification.GOOD]
    plt.plot(np.arange(len(good_df)), np.full(len(good_df), good_df[value].mean()),
             color='black', linestyle='--', figure=figure)

    bad_df: pd.DataFrame = str_dev.loc[str_dev['image_class']
                                       == Classification.BAD]
    plt.plot(np.arange(len(bad_df)) + len(good_df), np.full(len(bad_df), bad_df[value].mean()),
             color='black', linestyle='--', figure=figure)

    ugly_df: pd.DataFrame = str_dev.loc[str_dev['image_class']
                                        == Classification.UGLY]
    plt.plot(np.arange(len(ugly_df)) + len(good_df) + len(bad_df), np.full(len(ugly_df), ugly_df[value].mean()),
             color='black', linestyle='--', label='Mean value', figure=figure)


def standard_deviation(df: pd.DataFrame, rating_array: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates standard deviation of 2 (2.3% - 97,7%)
    In total about 95% values are kept and 5% thrown
    """
    df: pd.DataFrame = df[np.abs(
        stats.zscore(rating_array, nan_policy='omit')) < 2]
    return df


def save_fig(dictionary: Dict, path: Path, ext: str) -> None:
    """
    Saves figures stored in the dictionary
    """

    for variant in dictionary:
        fname = path / f"{variant}{ext}"
        dictionary[variant].savefig(fname)


def main(args: argparse.ArgumentParser) -> None:
    """
    Simple script for generating plots from log.json
    As the SD27 dataset was graded, we can compare those two
    gradings with this script
    """

    log_file_path: Path = args.sdir
    output_path: Path = args.ddir

    df = json2pandas(log_file_path)

    figures = {}

    # --------------------------------------------------------------------

    rating_array = df['minutiae_points']
    minutiae_points_df = standard_deviation(df, rating_array)

    minutiae_point_fig: plt.Figure = plt.figure(figsize=(22, 5), dpi=150)

    for index, row in minutiae_points_df.iterrows():
        plt.plot(index, row['minutiae_points'], marker="o",
                 color=row['color'], figure=minutiae_point_fig)

    plot_trendline(rating_array, minutiae_point_fig)

    plot_metadata('Minutia points', 'Fingeprint number',
                  'Number of minutiae', minutiae_point_fig)
    figures['minutiae_points_rating'] = minutiae_point_fig

    # --------------------------------------------------------------------

    rating_array = df['rmse']
    rmse_df = standard_deviation(df, rating_array)

    rmse_fig: plt.Figure = plt.figure(figsize=(22, 5), dpi=150)

    for index, row in rmse_df.iterrows():
        plt.plot(index, row['rmse'], marker="o",
                 color=row['color'], figure=rmse_fig)

    plot_trendline(rating_array, rmse_fig)

    plot_metadata('Root Mean Square Error', 'Fingeprint number',
                  'Root Mean Square Error value', rmse_fig)
    figures['rmse_rating'] = rmse_fig

    # --------------------------------------------------------------------

    rating_array = df['michelson']
    michelson_df = standard_deviation(df, rating_array)

    michelson_fig: plt.Figure = plt.figure(figsize=(22, 5), dpi=150)

    for index, row in michelson_df.iterrows():
        plt.plot(index, row['michelson'], marker="o",
                 color=row['color'], figure=michelson_fig)

    plot_trendline(rating_array, michelson_fig)

    plot_metadata('Michelson\'s contrast', 'Fingeprint number',
                  'Contrast value', michelson_fig)
    figures['michelson_rating'] = michelson_fig

    # --------------------------------------------------------------------

    rating_array = df['papilary_ridges']
    num_rig_df = standard_deviation(df, rating_array)

    num_rig_fig: plt.Figure = plt.figure(figsize=(22, 5), dpi=150)

    for index, row in num_rig_df.iterrows():
        plt.plot(index, row['papilary_ridges'], marker="o",
                 color=row['color'], figure=num_rig_fig)

    plot_trendline(rating_array, num_rig_fig)

    plot_metadata('Number of ridges', 'Fingeprint number',
                  'Number of ridges', num_rig_fig)
    figures['number_of_ridges_rating'] = num_rig_fig

    # --------------------------------------------------------------------

    rating_array = df['sin_s']
    sin_s_df = standard_deviation(df, rating_array)

    sin_s_fig: plt.Figure = plt.figure(figsize=(22, 5), dpi=150)

    for index, row in sin_s_df.iterrows():
        plt.plot(index, row['sin_s'], marker="o",
                 color=row['color'], figure=sin_s_fig)

    plot_zero_line(sin_s_df, 'sin_s', sin_s_fig)

    plt.axhline(y=0, color='r', linestyle='-',
                label='Ideal value', figure=sin_s_fig)
    plot_metadata('Sinusoidal Crosscut AEC', 'Fingeprint number',
                  'Sinusoidal deviance', sin_s_fig)
    figures['sinusoidal_similarity_rating'] = sin_s_fig

    # --------------------------------------------------------------------

    rating_array = df['thick']
    thick_df = standard_deviation(df, rating_array)

    thick_fig: plt.Figure = plt.figure(figsize=(22, 5), dpi=150)

    for index, row in thick_df.iterrows():
        plt.plot(index, row['thick'], marker="o",
                 color=row['color'], figure=thick_fig)

    plot_zero_line(thick_df, 'thick', thick_fig)

    plt.axhline(y=0, color='r', linestyle='-',
                label='Ideal value', figure=thick_fig)
    plot_metadata('Thickness AEC', 'Fingeprint number',
                  'Thickness deviance', thick_fig)
    figures['thickness_rating'] = thick_fig

    # --------------------------------------------------------------------

    rating_array = df['sin_s_core']
    sin_s_core_df = standard_deviation(df, rating_array)

    sin_s_core_fig: plt.Figure = plt.figure(figsize=(22, 5), dpi=150)

    for index, row in sin_s_core_df.iterrows():
        plt.plot(index, row['sin_s_core'], marker="o",
                 color=row['color'], figure=sin_s_core_fig)

    plot_zero_line(sin_s_core_df, 'sin_s_core', sin_s_core_fig)

    plt.axhline(y=0, color='r', linestyle='-',
                label='Ideal value', figure=sin_s_core_fig)
    plot_metadata('Sinusoidal Crosscut Core AEC', 'Fingeprint number',
                  'Sinusoidal deviance', sin_s_core_fig)
    figures['sinusoidal_similarity_core_rating'] = sin_s_core_fig

    # --------------------------------------------------------------------

    rating_array = df['thick_core']
    thick_core_df = standard_deviation(df, rating_array)

    thick_core_fig: plt.Figure = plt.figure(figsize=(22, 5), dpi=150)

    for index, row in thick_core_df.iterrows():
        plt.plot(index, row['thick_core'], marker="o",
                 color=row['color'], figure=thick_core_fig)

    plot_zero_line(thick_core_df, 'thick_core', thick_core_fig)

    plt.axhline(y=0, color='r', linestyle='-',
                label='Ideal value', figure=thick_core_fig)
    plot_metadata('Thickness Core AEC', 'Fingeprint number',
                  'Thickness deviance', thick_core_fig)
    figures['thickness_core_rating'] = thick_core_fig

    # --------------------------------------------------------------------

    rating_array = df['sin_s_gray']
    sin_s_gray_df = standard_deviation(df, rating_array)

    sin_s_gray_fig: plt.Figure = plt.figure(figsize=(22, 5), dpi=150)

    for index, row in sin_s_gray_df.iterrows():
        plt.plot(index, row['sin_s_core'], marker="o",
                 color=row['color'], figure=sin_s_gray_fig)

    plot_zero_line(sin_s_gray_df, 'sin_s_gray', sin_s_gray_fig)

    plt.axhline(y=0, color='r', linestyle='-',
                label='Ideal value', figure=sin_s_gray_fig)
    plot_metadata('Sinusoidal Crosscut Grayscale', 'Fingeprint number',
                  'Sinusoidal deviance', sin_s_gray_fig)
    figures['sinusoidal_similarity_gray_rating'] = sin_s_gray_fig

    # --------------------------------------------------------------------

    rating_array = df['thick_gray']
    thick_gray_df = standard_deviation(df, rating_array)

    thick_gray_fig: plt.Figure = plt.figure(figsize=(22, 5), dpi=150)

    for index, row in thick_gray_df.iterrows():
        plt.plot(index, row['thick_gray'], marker="o",
                 color=row['color'], figure=thick_gray_fig)

    plot_zero_line(thick_gray_df, 'thick_gray', thick_gray_fig)

    plt.axhline(y=0, color='r', linestyle='-',
                label='Ideal value', figure=thick_gray_fig)
    plot_metadata('Thickness Grayscale', 'Fingeprint number',
                  'Thickness deviance', thick_gray_fig)
    figures['thickness_gray_rating'] = thick_gray_fig

    # --------------------------------------------------------------------

    rating_array = df['sin_s_core_gray']
    sin_s_core_gray_df = standard_deviation(df, rating_array)

    sin_s_core_gray_fig: plt.Figure = plt.figure(figsize=(22, 5), dpi=150)

    for index, row in sin_s_core_gray_df.iterrows():
        plt.plot(index, row['sin_s_core_gray'], marker="o",
                 color=row['color'], figure=sin_s_core_gray_fig)

    plot_zero_line(
        sin_s_core_gray_df, 'sin_s_core_gray', sin_s_core_gray_fig
    )

    plt.axhline(y=0, color='r', linestyle='-',
                label='Ideal value', figure=sin_s_core_gray_fig)
    plot_metadata('Sinusoidal Crosscut Core Grayscale', 'Fingeprint number',
                  'Sinusoidal deviance', sin_s_core_gray_fig)
    figures['sinusoidal_similarity_core_gray_rating'] = sin_s_core_gray_fig

    # --------------------------------------------------------------------

    rating_array = df['thick_core_gray']
    thick_core_gray_df = standard_deviation(df, rating_array)

    thick_core_gray_fig: plt.Figure = plt.figure(figsize=(22, 5), dpi=150)

    for index, row in thick_core_gray_df.iterrows():
        plt.plot(index, row['thick_core_gray'], marker="o",
                 color=row['color'], figure=thick_core_gray_fig)

    plot_zero_line(
        thick_core_gray_df, 'thick_core_gray', thick_core_gray_fig
    )

    plt.axhline(y=0, color='r', linestyle='-',
                label='Ideal value', figure=thick_core_gray_fig)
    plot_metadata('Thickness Core Grayscale', 'Fingeprint number',
                  'Thickness deviance', thick_core_gray_fig)
    figures['thickness_core_gray_rating'] = thick_core_gray_fig

    # --------------------------------------------------------------------

    save_fig(figures, output_path, '.jpeg')


if __name__ == "__main__":
    args = argument_parse()
    main(args)
