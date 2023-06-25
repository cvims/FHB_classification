#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Organization  : Technische Hochschule Ingolstadt
# Institution   : Computer Vision for Intelligent Mobility Systems (CVIMS)
# Created By    : Dominik Rößle
# Created Date  : Wed Nov 02 2022
# =============================================================================
"""
File description.
"""
# =============================================================================
# Imports
# =============================================================================
from torchvision.transforms import transforms
from typing import List
import numpy as np


data_characteristics = {
    2020: {
        'height_mean': 3039.075934579439,
        'width_mean': 909.3528037383178,
        'ch_mean': [0.43141420093571287, 0.48390027344432324, 0.38983213420029555],
        'ch_std': [0.3089539263768662, 0.3061668088248115, 0.29685788262067875]
    },
    2021: {
        'height_mean': 2414.6846394984327,
        'width_mean': 894.3253918495298,
        'ch_mean': [0.46229406980215765, 0.5221872278336893, 0.3918821595316316],
        'ch_std': [0.3099188808326271, 0.299705719135797, 0.29576527208858083]
    },
    2022: {
        'height_mean': 3208.7297830374755,
        'width_mean': 1507.32741617357,
        'ch_mean': [0.5336078369903198, 0.5066206422819128, 0.37175902280974665],
        'ch_std': [0.277231204060943, 0.28374415653425855, 0.2805299531810641]
    },
    'all_years': {
        'height_mean': 2731.472954699121,
        'width_mean': 1003.7423935091277,
        'ch_mean': [0.4835300530634593, 0.5161176709301399, 0.39255822057793854],
        'ch_std': [0.29858720854908294, 0.2927398784147614, 0.29034040924473015]
    }
}


def merge_characteristics(characteristics: List):
    height_mean = []
    width_mean = []
    ch_mean = []
    ch_std = []
    for characteristic in characteristics:
        height_mean.append(characteristic['height_mean'])
        width_mean.append(characteristic['width_mean'])
        ch_mean.append(characteristic['ch_mean'])
        ch_std.append(characteristic['ch_std'])
    
    # mean the entries
    return {
        'height_mean': sum(height_mean) / len(height_mean),
        'width_mean': sum(width_mean) / len(width_mean),
        'ch_mean': np.sum(ch_mean, axis=0) / len(ch_mean),
        'ch_std': np.sum(ch_std, axis=0) / len(ch_std)
    }


def get_characteristics(data_year: int or str):
    if data_year and str(data_year).isdigit():
        return data_characteristics[int(data_year)]
    elif data_year and isinstance(data_year, List):
        characteristics = []
        for year in data_year:
            if str(year).isdigit():
                characteristics.append(data_characteristics[int(year)])
            else:
                characteristics.append(data_characteristics['all_years'])

        return merge_characteristics(characteristics=characteristics)
    else:
        # from all years together
        return data_characteristics['all_years']


def train_transforms(data_year: int or str):
    characteristics = get_characteristics(data_year)

    means, stds = characteristics['ch_mean'], characteristics['ch_std']

    h_mean, w_mean = int(characteristics['height_mean']), int(characteristics['width_mean'])
    # rand_crop (keep 90%)
    keep_perc = 0.90
    # first increase the height and width and reduce the size afterwards to the initial size (cropped)
    h_mean, w_mean = int((1 / keep_perc) * h_mean), int((1 / keep_perc) * w_mean)
    # random_crop_h = int(h_mean * keep_perc)
    # random_crop_w = int(w_mean * keep_perc)
    random_crop_h = int(characteristics['height_mean'])
    random_crop_w = int(characteristics['width_mean'])

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=(h_mean, w_mean)),
            transforms.RandomRotation(degrees=2.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(size=(random_crop_h, random_crop_w)),
            # transforms.RandomVerticalFlip(p=0.5),
            # transforms.ColorJitter(
            #     brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05
            # ),
            transforms.Normalize(mean=means, std=stds)
        ]
    )


def val_transforms(data_year: int or str):
    characteristics = get_characteristics(data_year)
    means, stds = characteristics['ch_mean'], characteristics['ch_std']

    h_mean, w_mean = int(characteristics['height_mean']), int(characteristics['width_mean'])

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=(h_mean, w_mean)),
            transforms.Normalize(mean=means, std=stds)
        ]
    )


def test_transforms(data_year: int or str):
    characteristics = get_characteristics(data_year)
    means, stds = characteristics['ch_mean'], characteristics['ch_std']

    h_mean, w_mean = int(characteristics['height_mean']), int(characteristics['width_mean'])

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=(h_mean, w_mean)),
            transforms.Normalize(mean=means, std=stds)
        ]
    )