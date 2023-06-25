#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Organization  : Technische Hochschule Ingolstadt
# Institution   : Computer Vision for Intelligent Mobility Systems (CVIMS)
# Created By    : Author
# Created Date  : Wed October 26 2022
# Latest Update : Wed October 26 2022
# =============================================================================
"""
Fusarium data loader.
"""
# =============================================================================
# Imports
# =============================================================================
import os
import copy
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Any, Callable, Dict, List, Tuple
import functools
import numpy as np
from src.utils.utils import load_annotation_file


class _FusariumDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        labels: List[str],
        specific_labels: List[int] = None,
        include_label_borders: bool = False,
        transform: Callable = None,
        floating_point_labels: bool = False,
    ) -> None:
        """
        Fusarium dataset
        :param image_paths:
        :param labels:
        :param specific_labels:
        :param include_label_borders: Includes the borders for the labels if specific labels
        is set to a sequence. Borders means including all the lower labels to one label and all the
        higher labels building one label.
        :param transform:
        :param floating_point_labels:
        """
        super().__init__()
        assert len(image_paths) == len(labels), 'Image paths and labels must have same lengths.'

        self._image_paths = image_paths
        if isinstance(self._image_paths, set):
            self._image_paths = list(self._image_paths)

        self._labels = labels
        if isinstance(self._labels, set):
            self._labels = list(self._labels)

        self.specific_labels = specific_labels
        self.transform = transform
        self.floating_point_labels = floating_point_labels
        self.include_label_borders = include_label_borders

        if include_label_borders:
            assert self.specific_labels, 'Please provide specific labels when using include_label_borders.'
        
            # make sure specific labels are ordered and without gaps
            _min_s = min(specific_labels)
            assert all([i == s_l - _min_s for i, s_l in enumerate(specific_labels)])
    
        if specific_labels:
            self._image_paths, self._labels = self.load_only_specifics(
                specific_labels=self.specific_labels,
                image_paths=self._image_paths,
                labels=self._labels,
                include_label_borders=self.include_label_borders
            )

        # start with label 0 ascending (for labels - no gaps)
        if floating_point_labels:
            # we consider the labels as a range starting from 0 (we assume that the labels start from 0)
            _min = min(self._labels)
            self._internal_labels = lambda key: key - _min
        else:
            _internal_labels = {key: i for i, key in enumerate(sorted(set(self._labels)))}
            self._internal_labels = lambda key: _internal_labels[key]
        
        # images per label
        self._calculate_label_lengths()
    
    def _calculate_label_lengths(self) -> None:
        self._label_lengths = {}
        for label in self._labels:
            if label not in self._label_lengths:
                self._label_lengths[label] = 1
            else:
                self._label_lengths[label] = self._label_lengths[label] + 1
        
        self._label_lengths = {label: self._label_lengths[label] for label in sorted(set(self._label_lengths.keys()))}
    
    def unique_labels(self) -> List[int]:
        return sorted(list(set(self._labels)))

    def _label_counts(self) -> Dict[int, int]:
        """
        Returns a dictionary with the unique labels as key and the amount of images per label in the dataset as value.
        """
        return self._label_lengths
    
    @staticmethod
    def load_only_specifics(
        specific_labels: List[int], image_paths: List[str], labels: List[int],
        include_label_borders: bool = False
    ) -> Tuple[List[str], List[int]]:
        """
        Loads specifics labels from the dataset and changes the images and labels variables.
        """

        image_paths = copy.deepcopy(image_paths)
        labels = copy.deepcopy(labels)
        floating_point_labels = all([not isinstance(label, int) for label in labels])

        to_delete = []
        to_delete_vals = []
        for i, label in enumerate(labels):
            if floating_point_labels:
                eps = 1e-5
                # we also include values from min - 0.5 and max + 0.5 to make it even to the round function of classification tasks
                if label < (min(specific_labels) - eps - 0.5) or label > (max(specific_labels) + eps + 0.5):
                    to_delete.append(i)
                    to_delete_vals.append(label)
            else:
                if label not in specific_labels:
                    to_delete.append(i)
                    to_delete_vals.append(label)
        
        _min_sl, _max_sl = min(specific_labels), max(specific_labels)
        for delete_index in reversed(to_delete):
            if include_label_borders:
                if labels[delete_index] > _max_sl:
                    # change to _max_sl + 1
                    labels[delete_index] = _max_sl + 1
                else:
                    labels[delete_index] = _min_sl - 1
            else:
                del image_paths[delete_index]
                del labels[delete_index]
        
        return image_paths, labels

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx) -> Tuple[Any, torch.Tensor]:
        image = Image.open(self._image_paths[idx])
        label = torch.tensor(self._internal_labels(self._labels[idx]))

        if self.transform:
            image = self.transform(image)

        return image, label


class FusariumDataset(_FusariumDataset):
    def __init__(
        self,
        annotation_file_name: str,
        data_dir: str,
        data_year: str or int = None,
        camera_name: str = None,
        specific_labels: List[int] = None,
        include_label_borders: bool = False,
        transform: Callable = None
    ) -> None:
        """
        Dataset to load the Fusarium images with annotations.
        :param camera_name: Name of camera directory inside a data year directory of dataset.
        :param annotation_file_name: Annotation file name.
        :param start_dir: By default the parent dir to search for images with the given annotation file.
        :param specific_labels: Only for loading specific labels of the Fusarium dataset.
        :param transform: torch transformations to process images.
        :param include_label_borders:
        """
        assert os.path.isdir(data_dir), 'Start directory not found.'        

        annotation_file_path = os.path.join(data_dir, annotation_file_name)

        data_year = str(data_year) if data_year and str(data_year).isdigit() else ''
        camera_name = camera_name if camera_name and camera_name != 'all_cameras' else ''

        # add expert annotations tag to path
        annotation_file_path = os.path.join(
            data_dir,
            'expert_annotations',
            data_year,
            camera_name,
            annotation_file_name)
        
        data_dir = os.path.join(data_dir, data_year, camera_name)

        assert os.path.isfile(annotation_file_path), f'Path to annotation file not found, got {annotation_file_path}.'

        image_paths, labels = load_annotation_file(file_path=annotation_file_path, start_dir=data_dir)

        super().__init__(
            image_paths=image_paths,
            labels=labels,
            specific_labels=specific_labels,
            transform=transform,
            include_label_borders=include_label_borders
        )


class FusariumDatasetMerger(_FusariumDataset):
    def __init__(
        self,
        fusarium_datasets: List[FusariumDataset],
        specific_labels: List[int] = None,
        include_label_borders: bool = False,
        transform: Callable = None,
        label_mode: str = 'mean',
        keep_mode: str = 'intersection',  # intersection or union
        round_labels: bool = False,
        floating_point_labels: bool = False
    ) -> None:
        """
        Merges the labels of multiple fusarium datasets and
        """
        image_paths, labels = self._fuse_datasets(
            fusarium_datasets=fusarium_datasets,
            keep_mode=keep_mode,
            label_mode=label_mode,
            round_labels=round_labels
        )

        # we have to fake the labels so that "specific labels can take place"
        super().__init__(
            image_paths=image_paths,
            labels=labels,
            specific_labels=specific_labels,
            transform=transform,
            floating_point_labels=floating_point_labels,
            include_label_borders=include_label_borders
        )

        # override labels here with the correct labels
        self._labels

    def _merge_data(self, fusarium_datasets: List[_FusariumDataset], keep_mode: str) -> List[str]:
        valid_image_paths = None
        if keep_mode == 'intersection':
            valid_image_paths = functools.reduce(lambda f, s: f.intersection(s), [set(fusarium_dataset._image_paths) for fusarium_dataset in fusarium_datasets])
        elif keep_mode == 'union':
            valid_image_paths = functools.reduce(lambda f, s: f.union(s), [set(fusarium_dataset._image_paths) for fusarium_dataset in fusarium_datasets])
        else:
            raise NotImplementedError(f'Method {keep_mode} not implemented.')
        
        overlapping_indices = []
        if valid_image_paths:
            # find indices of overlapping valid image paths
            for fusarium_dataset in fusarium_datasets:
                ds_valids = []
                i_paths = set(fusarium_dataset._image_paths)
                for image_path in valid_image_paths:
                    if image_path in i_paths:
                        ds_valids.append(1)
                    else:
                        ds_valids.append(0)
                overlapping_indices.append(ds_valids)

        elem_wise_overlap = []
        for i, valid_in_ds in enumerate(np.array(overlapping_indices).T):
            valid_ids = np.where(valid_in_ds == 1)[0]

            # elem_wise_overlap.
            # First param is the valid image paths index
            # Second param is for the datasets which overlap (in ordner of the passed datasets)
            # Third param is for the standalone datasets (if the label is not shared)
            # Param two and three are exclusive to each other
            if not np.any(valid_ids) or np.sum(valid_in_ds) <= 1.0:
                elem_wise_overlap.append((i, None, valid_ids[0]))
            else:
                elem_wise_overlap.append((i, valid_ids.tolist(), None))

        return list(valid_image_paths), elem_wise_overlap
    
    def _merge_labels(self, fusarium_datasets: List[_FusariumDataset], valid_image_paths: List[str], overlap_indices: List, label_mode: str, round_labels: bool = False) -> List[float]:
        new_labels = None
        # keeps the dataset indices in relation to the valid paths with image path index. This is just for easy accessing the data
        valid_image_dataset_index_dict = {
            d_i: {image_path: label for image_path, label in zip(dataset._image_paths, dataset._labels)} for d_i, dataset in enumerate(fusarium_datasets)
        }

        if label_mode == 'mean':
            # only mean those who are overlapping
            new_labels = []
            for overlap in overlap_indices:
                image_path = valid_image_paths[overlap[0]]
                overlapping_dataset_indices = overlap[1]
                standalone_dataset_id = overlap[2]

                if overlapping_dataset_indices:
                    labels = [valid_image_dataset_index_dict[i][image_path] for i in overlapping_dataset_indices]
                    new_labels.append(np.mean(labels))
                
                if standalone_dataset_id is not None:
                    new_labels.append(valid_image_dataset_index_dict[standalone_dataset_id][image_path])

            new_labels = torch.as_tensor(new_labels).to(dtype=torch.float32)
        else:
            raise NotImplementedError(f'Method {self.keep_mode} not implemented.')
        
        result = None
        if round_labels:
            result = torch.round(new_labels).to(dtype=torch.uint8)
        else:
            result = new_labels
        
        if isinstance(result, torch.Tensor):
            return result.tolist()
        else:
            return result

    def _fuse_datasets(
        self,
        fusarium_datasets: List[_FusariumDataset],
        keep_mode: str,
        label_mode: str,
        round_labels: bool = False
    ) -> Tuple[List[str], List[int]]:
        valid_image_paths, overlap_indices = self._merge_data(
            fusarium_datasets=fusarium_datasets,
            keep_mode=keep_mode
        )
        labels = self._merge_labels(
            fusarium_datasets=fusarium_datasets,
            valid_image_paths=valid_image_paths,
            overlap_indices=overlap_indices,
            label_mode=label_mode,
            round_labels=round_labels
        )

        return valid_image_paths, labels
