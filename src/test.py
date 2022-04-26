#!/usr/bin/env python3.7
# coding: utf-8

import argparse
from pathlib import Path
from enum import IntEnum
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
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

    df['rmse_ridge'] = contrast['rmse_ridge']
    df['rmse_valley'] = contrast['rmse_valley']
    df['rmse_ratio'] = contrast['rmse_ratio']
    df['color_ratio'] = contrast['color_ratio']
    df['col_diff_ridge'] = contrast['col_diff_ridge']
    df['col_diff_valley'] = contrast['col_diff_valley']
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


def plot_trendline(first: pd.DataFrame, second: pd.DataFrame, figure: plt.Figure) -> None:
    """
    Plot trendline based on the least squares polynomial fit 
    of the first degree
    """
    first = first.dropna()
    second = second.dropna()
    z = np.polyfit(first, second, 1)
    p = np.poly1d(z)
    plt.plot(first, p(second),
             "r--", figure=figure, label="Trendline")


def plot_metadata(title, x_label: str, y_label: str, figure: plt.Figure) -> None:
    """
    Data for the plot to generate nice charts
    """

    color_legend = {'Good': u'lightgreen', 'Bad': u'sandybrown',
                    'Ugly': u'indianred'}  # , 'Unknown': u'black'

    plt.title(title)
    plt.xlabel(x_label, fontsize='small', figure=figure)
    plt.ylabel(y_label, fontsize='small', figure=figure)
    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='')
               for color in color_legend.values()]
    plt.legend(markers, color_legend.keys(), numpoints=1, fontsize='small')
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
    🦊 🐺 🦌 🐕
    """

    log_file_path: Path = args.sdir
    output_path: Path = args.ddir

    df = json2pandas(log_file_path)

    figures = {}

    def prepare_data(df, type: str, title: str, y_label: str, line=None):
        mp_df = df[[type, 'image_class']]

        good = mp_df.loc[mp_df['image_class'] == Classification.GOOD]
        bad = mp_df.loc[mp_df['image_class'] == Classification.BAD]
        ugly = mp_df.loc[mp_df['image_class'] == Classification.UGLY]

        fig = plt.figure(figsize=(5, 3))
        data = [
            standard_deviation(good, good[type])[type],
            standard_deviation(bad, bad[type])[type],
            standard_deviation(ugly, ugly[type])[type]
        ]
        test = plt.boxplot(data, patch_artist=True)
        plt.xticks([1, 2, 3], ["Good", "Bad", "Ugly"], figure=fig)

        colors = [u'lightgreen', u'sandybrown', u'indianred']
        for patch, color in zip(test['boxes'], colors):
            patch.set_facecolor(color)
        for median in test['medians']:
            median.set_color('black')

        if line is not None:
            plt.axhline(y=1, color='r', linestyle='--',
                        label='Ideal value', figure=fig
                        )

        plt.title(title, figure=fig)
        plt.ylabel(y_label, fontsize='small', figure=fig)
        plt.close()

        return fig

    # df = df.dropna()
    # df = df.sort_values(by='color_ratio', ascending=False)
    # # df = df[df["sin_s"] > -0.1]
    # # df = df[df["sin_s"] < 1.02]
    # print(df[["image","color_ratio", "col_diff_ridge", "col_diff_valley"]])
    # exit(0)

    figures['minutiae_points_rating'] = prepare_data(
        df, 'minutiae_points', 'Minutiae points', 'Number of minutiae'
    )

    figures['number_of_ridges_rating'] = prepare_data(
        df, 'papilary_ridges', 'Number of ridges', 'Ridge count'
    )

    figures['michelson_rating'] = prepare_data(
        df, 'michelson', 'Michelson\'s contrast', 'Contrast value'
    )

    figures['col_diff_ridge_rating'] = prepare_data(
        df, 'col_diff_ridge', 'Color difference of ridges', 'Mean color difference'
    )

    figures['col_diff_valley_rating'] = prepare_data(
        df, 'col_diff_valley', 'Color difference of valleys', 'Mean color difference'
    )

    figures['color_ratio_rating'] = prepare_data(
        df, 'color_ratio', 'Color difference ratio', 'Color difference', line=1
    )

    figures['rmse_ratio_rating'] = prepare_data(
        df, 'rmse_ratio', 'Root Mean Square Error ratio', 'Root Mean Square Error value'
    )

    figures['rmse_valley_rating'] = prepare_data(
        df, 'rmse_valley', 'Root Mean Square Error valley', 'Root Mean Square Error value'
    )

    figures['rmse_ridge_rating'] = prepare_data(
        df, 'rmse_ridge', 'Root Mean Square Error ridges', 'Root Mean Square Error value'
    )

    # --------------------------------------------------------------------

    figures['sinusoidal_similarity_rating'] = prepare_data(
        df, 'sin_s', '', 'Sinusoidal deviance', line=0)
    figures['thickness_rating'] = prepare_data(
        df, 'thick', '', 'Thickness deviance', line=0)
    figures['sinusoidal_similarity_core_rating'] = prepare_data(
        df, 'sin_s_core', '', 'Sinusoidal deviance', line=0)
    figures['thickness_core_rating'] = prepare_data(
        df, 'thick_core', '', 'Thickness deviance', line=0)

    figures['sinusoidal_similarity_gray_rating'] = prepare_data(
        df, 'sin_s_gray', '', 'Sinusoidal deviance', line=0
    )
    figures['thickness_gray_rating'] = prepare_data(
        df, 'thick_gray', '', 'Thickness deviance', line=0
    )
    figures['sinusoidal_similarity_core_gray_rating'] = prepare_data(
        df, 'sin_s_core_gray', '', 'Sinusoidal deviance', line=0
    )
    figures['thickness_core_gray_rating'] = prepare_data(
        df, 'thick_core_gray', '', 'Thickness deviance', line=0
    )

    # --------------------------------------------------------------------

    def line_chart(df, grade: Classification, title: str, x_label: str, y_label: str):

        mp_df = df[['col_diff_ridge', 'col_diff_valley', 'image_class', 'color']]

        classified = mp_df.loc[mp_df['image_class'] == grade]
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)

        major_ticks = np.arange(0, 1.1, 0.1)

        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)

        ax.grid(which='both')
        ax.grid(which='major', alpha=0.5, linestyle=':',
                color='gray', linewidth=1)

        for _, row in classified.iterrows():
            plt.plot(row['col_diff_valley'], row['col_diff_ridge'], marker="o",
                     color=row['color'], figure=fig)

        plt.plot(np.linspace(0, 1, 2), color='blue',
                 linestyle='-', label='Ideal ratio', figure=fig
                 )
        plt.axhline(y=0.5, color='black', linestyle='-',
                    figure=fig, linewidth=1
                    )
        plt.axvline(x=0.5, color='black', linestyle='-',
                    label='Box', figure=fig, linewidth=1
                    )

        plt.title(title, figure=fig)
        plt.xlabel(x_label, fontsize='small', figure=fig)
        plt.ylabel(y_label, fontsize='small', figure=fig)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.close()

        return fig

    figures['ratio_good_rating'] = line_chart(
        df, Classification.GOOD, 'Color difference ratio – Good', 'Valley mean color', 'Ridge mean color')
    figures['ratio_bad_rating'] = line_chart(
        df, Classification.BAD, 'Color difference ratio – Bad', 'Valley mean color', 'Ridge mean color')
    figures['ratio_ugly_rating'] = line_chart(
        df, Classification.UGLY, 'Color difference ratio – Ugly', 'Valley mean color', 'Ridge mean color')

    save_fig(figures, output_path, '.pgf')


def mathplotlib_settings():
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })


if __name__ == "__main__":
    args = argument_parse()
    mathplotlib_settings()
    main(args)
