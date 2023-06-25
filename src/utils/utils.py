#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Organization  : Technische Hochschule Ingolstadt
# Institution   : Computer Vision for Intelligent Mobility Systems (CVIMS)
# Created By    : Dominik Rößle
# Created Date  : Wednesday April 20 hh:mm:ss GMT 2022
# Latest Update : Wednesday April 20 hh:mm:ss GMT 2022
# =============================================================================
"""
Scale definitions of the dataset. Optimal width and height calculation
"""
# =============================================================================
# Imports
# =============================================================================
import os
from typing import Tuple, List, Callable

import tqdm as tqdm
from PIL import Image
import numpy as np


def load_image(image_file_path: str) -> np.ndarray:
    """
    """
    img = Image.open(image_file_path)
    return np.asarray(img)


def image_opener_iterator(search_dir: str, header: str, content: List[str]):
    """
    """
    # find image urls
    headers = header.split(';')
    image_url_index = header.index('image_url')
    # user_scores_index = header.index('user_scores')

    if image_url_index < 0: # or user_scores_index < 0:
        raise Exception(f'Invalid annotation file. Required headers are "image_url" and "user_scores", got {headers}.')

    for line in tqdm.tqdm(content, 'Loading images for preprocessing'):
        elements = line.split(';')
        image_url = elements[image_url_index]
        # user_score = elements[user_scores_index]

        file_path = os.path.join(search_dir, image_url)
        if not os.path.isfile(file_path):
            print(f'Cannot find file at path {file_path}.')
            continue
    
        yield load_image(file_path), file_path, line


def calculate_height_width(
    image_heights: List[int], image_widths: List[int]
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Calculates one height and width representation for a set of different sized images
    :param image_heights: List of image heights
    :param image_widths: List of image widths
    :return: Tuple mean(height, width), Tuple min(height, width), Tuple max(height, width)
    """

    mean = round(sum(image_heights) / len(image_heights)), round(sum(image_widths) / len(image_widths))
    _min = min(image_heights), min(image_widths)
    _max = max(image_heights), max(image_widths)

    return mean, _min, _max


def remove_black_pixels_by_amount(search_dir: str, header: str, content: List[str],
                                  max_black_pixel_percentage: float = 0.05):
    """
    """
    new_content = []
    for img, _, line in image_opener_iterator(
        search_dir=search_dir, header=header, content=content
    ):
        flattened_img = img.flatten()
        max_black_pixels = (1 - max_black_pixel_percentage) * flattened_img.size

        if not np.count_nonzero(flattened_img) < max_black_pixels:
            new_content.append(line)
    
    removed_lines = len(content) - len(new_content)
    print(f'Removed {removed_lines}/{len(content)} because of a too high amount of black pixels.')
    
    return new_content


def remove_text_characters(text: str):
    """
    Removes unnecessary characters for reading the annotations.
    :param text: Parameter to search for replacements.
    """
    text = str(text).replace('"', '')
    text = text.replace("\r", "")
    text = text.replace("\n", "")

    # is float or int?
    if len(text) > 0 and all(char.isdigit() or char == '.' for char in text):
        if '.' in text:
            # float
            text = float(text)
        else:
            text = int(text)

    return text


def load_annotation_file(file_path: str, start_dir: str = None) -> Tuple[List[str], List[int]]:
    if start_dir is None:
        start_dir = os.path.join(*os.path.split(file_path)[:-1])

    delimitter = ';'
    image_paths = []
    image_annotations = []
    
    # open annotation file
    with open(file_path, 'r') as annotation_file:
        # header
        header = annotation_file.readline()
        header = remove_text_characters(header)

        headers = list(header.split(delimitter))
        
        image_url_index = headers.index('image_url')
        annotation_index = headers.index('user_scores')

        assert image_url_index >= 0 and annotation_index >= 0, 'Could not find "image_url" or "user_scores" header, both are required.'

        line = annotation_file.readline()
        while line:
            line = remove_text_characters(line)
            l_split = line.split(delimitter)
            image_path = l_split[image_url_index]
            annotation = l_split[annotation_index]

            if not annotation.isdigit() or int(annotation) <= 0:
                line = annotation_file.readline()
                continue
            
            full_image_path = os.path.join(start_dir, image_path)
            if not os.path.isfile(full_image_path):
                print(f'Cannot find image at path: {full_image_path}.')
                line = annotation_file.readline()
                continue
        
            # add path
            image_paths.append(full_image_path)
            image_annotations.append(int(annotation))
                
            line = annotation_file.readline()
    
    return image_paths, image_annotations


def find_expert_annotation_dir(data_dir: str, data_year: str or int, camera_name: str = None):
    """
    :param data_dir: Data dir from where the relative annotation paths from
    the annotation file can find the image data
    """
    if data_year and str(data_year).isdigit():
        data_dir = os.path.join(data_dir, 'expert_annotations', str(data_year))
        if camera_name:
            data_dir = os.path.join(data_dir, camera_name)
    
    return data_dir


def _find_expert_annotation_files(data_dir: str, data_year: str, camera_name: str = None):
    """
    :param data_dir: Data dir from where the relative annotation paths from
    the annotation file can find the image data
    :param data_year: Information about the dataset year. None if all years combined
    :param camera_name: Camera folder inside data year folder
    """
    expert_annotations_dir = find_expert_annotation_dir(
        data_dir=data_dir, data_year=data_year, camera_name=camera_name
    )

    # search for text files only
    expert_annotations_files = [f for f in os.listdir(path=expert_annotations_dir) if f[-4:] == '.txt']

    return expert_annotations_files, expert_annotations_dir


def find_expert_annotation_files_originals(data_dir: str, data_year: str or int, camera_name: str = None):
    """
    :param data_dir: Data dir from where the relative annotation paths from
    the annotation file can find the image data
    :param data_year: Information about the dataset year. None if all years combined
    :param camera_name: Inside of data year directory.
    """
    # search for text files only
    expert_annotations, expert_annotations_dir = _find_expert_annotation_files(
        data_dir=data_dir,
        data_year=data_year,
        camera_name=camera_name
    )

    expert_annotations = [f for f in expert_annotations if 'train' not in f]
    expert_annotations = [f for f in expert_annotations if 'val' not in f]
    expert_annotations = [f for f in expert_annotations if 'test' not in f]

    # also remove all files which have more than 1 underscore in name
    expert_annotations = [f for f in expert_annotations if len(f.split('_')) == 2]

    return expert_annotations, expert_annotations_dir


def find_expert_annotation_files_by_tag(data_dir: str, data_year: str, file_name_tag: str, camera_name: str = None):
    """
    :param data_dir: Data dir from where the relative annotation paths from
    the annotation file can find the image data
    :param data_year: Information about the dataset year. None if all years combined
    :param test_annotation_file_name: Name addition of the test annotation file.
    """
    # search for text files only
    expert_annotations, expert_annotations_dir = _find_expert_annotation_files(
        data_dir=data_dir,
        data_year=data_year,
        camera_name=camera_name
    )

    # search for proper test_file names only
    expert_annotations = [f for f in expert_annotations if file_name_tag == f[-len(file_name_tag):]]

    # also remove all files which have more than 1 underscore in name (otherwise annotations.txt cannot be requested without getting train/val/test as well.)
    expert_names = set([f.split('_')[0] for f in expert_annotations])
    expert_annotations_filtered = ['_'.join([expert_name, file_name_tag]) for expert_name in expert_names]
    expert_annotations_filtered = [ex_ann for ex_ann in expert_annotations if ex_ann in expert_annotations]

    return expert_annotations_filtered, expert_annotations_dir


def read_annotations(annotation_file_path: str) -> Tuple[str, List[str]]:
    """
    Reads the annotation file and returns its header and its content.
    :param annotation_file_path: Path to annotation file.
    :returns: Tuple of header (1) and list of content (2)
    """
    assert os.path.isfile(annotation_file_path)
    assert annotation_file_path[-3:].lower() == 'txt'

    with open(annotation_file_path, newline='\n', mode='r') as annotation_file:
        # header
        header = annotation_file.readline()
        header = remove_text_characters(header)

        content = []
        line = annotation_file.readline()
        while line:
            line = remove_text_characters(line)
            content.append(line)
            line = annotation_file.readline()
    
    return header, content


def preprocess_data_annotations(annotation_file_path: str, data_dir: str, update_content_fn: Callable = None):
    delimitter = ';'

    if len(annotation_file_path) <= 4 or annotation_file_path[-4:] != '.txt':
        annotation_file_path = annotation_file_path + '.txt'

    header, content = read_annotations(
        annotation_file_path=annotation_file_path
    )

    # check and remove invalid annotations
    keep_content = []
    for line in content:
        _s = line.split(';')
        if not line or len(_s) != 2:
            continue

        rel_path = _s[0]
        label = _s[1]
        full_path = os.path.join(data_dir, rel_path)

        if not os.path.isfile(full_path):
            print(f'Cannot find image at path: {full_path}. Skipping ...')
            continue
        
        if not label.isdigit() or not int(label) > 0:
            print(f'Found invalid label: {label} for image path: {full_path}. Skipping ...')
            continue
        
        keep_content.append(line)
    
    content = keep_content

    headers = list(header.split(delimitter))

    if update_content_fn:
        content = update_content_fn(header, content, data_dir)
    
    return headers, content, delimitter


def iterate_dataset_years(root_path: str):
    year_dirs = [os.path.join(root_path, p) for p in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, p)) and os.path.basename(p).isdigit()]
    
    for data_dir in [root_path, *year_dirs]:
        yield data_dir


def iterate_dataset_cameras(root_path: str):
    assert os.path.basename(root_path).isdigit(), 'Please provide a dataset path with a specific year.'
    camera_dirs = [os.path.join(root_path, p) for p in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, p))]

    for data_dir in camera_dirs:
        yield data_dir


def write_txt(full_path: str, data_arr: List):
    with open(full_path, mode='w+') as ann_f:
        lines = '\n'.join(data_arr)
        ann_f.writelines(lines)
