#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Organization  : Technische Hochschule Ingolstadt
# Institution   : Computer Vision for Intelligent Mobility Systems (CVIMS)
# Created By    : Dominik Roessle
# Created Date  : Wed October 26 2022
# Latest Update : Wed October 26 2022
# =============================================================================
"""
Training script for Fusarium classification
"""
# =============================================================================
# Imports
# =============================================================================
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

from src.utils.helpers import initialize_data_loader, set_seed
from src.utils.dataset import _FusariumDataset, FusariumDataset, FusariumDatasetMerger
from src.utils.transforms import train_transforms, val_transforms

from src.utils.helpers import BeautifyDataLoaderIterations, calculate_accuracy, EarlyStoppingByLoss, NetworkOperator, get_model_trainable_parameter_count, save_model_information

from src.utils.models import load_model

from src.default_config import efficientnet_configuration


def tensorboard_accuracy_callback(loss_type: str):
    """
    Tensorboard hook to visualize the accuracy progress on tensorboard.
    """
    def additional_callback(tensorboard_writer: SummaryWriter, mode: str, epoch: int,
                            losses: torch.Tensor, model_outputs: torch.Tensor, targets: torch.Tensor):
        """
        See cvims package tensorboard hook parameter for NetworkOperate.operate
        """
        if loss_type == 'classification':
            accuracy = calculate_accuracy(
                model_outputs=model_outputs, targets=targets, calc_mean=True
            )
        else:
            model_outputs = torch.round(model_outputs).to(dtype=torch.uint8).squeeze()
            accuracy = (model_outputs == targets).to(dtype=torch.float32).mean().detach()

        tensorboard_writer.add_scalar('/'.join(['Accuracy', mode]), accuracy, epoch)
    
    return additional_callback


def create_hooks(
    train_data_loader: BeautifyDataLoaderIterations,
    val_data_loader: BeautifyDataLoaderIterations,
    lr_scheduler: _LRScheduler,
    loss_type: str,
    unfreeze_epoch: int,
    model: nn.Module
) -> Dict:
    """
    :param train_data_loader: Training set DataLoader in Beautify mode
    :param val_data_loader: Validation set DataLoader in Beautify mode
    :param lr_scheduler: Lr scheduler
    :param loss_type: 'classification' or 'regression'
    :returns: Dictionary containing all hook functions
    """

    def after_train_batch_hook(loss, outputs, targets, *args):
        """
        See cvims package hook parameters for NetworkOperate.operate
        """
        if train_data_loader.tqdm_bar:
            train_data_loader.tqdm_bar.set_postfix_str('Loss: {:3f}'.format(loss.item()))
    
    def after_val_batch_hook(loss, outputs, targets, *args):
        """
        See cvims package hook parameters for NetworkOperate.operate
        """
        if val_data_loader.tqdm_bar:
            val_data_loader.tqdm_bar.set_postfix_str('Loss: {:3f}'.format(loss.item()))
    
    unfrozen = False
    def after_epoch_hook(
        epoch: int,
        train_info: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        eval_info: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """
        See cvims package hook parameters for NetworkOperate.operate
        """
        # we only use it to set the scheduler correctly
        lr_scheduler.step()

        nonlocal unfrozen
        if epoch >= unfreeze_epoch and unfrozen is False:
            for param in model.parameters():
                param.requires_grad = True
            
            print('All weights unfrozen and trainable.')
            unfrozen = True


    def tensorboard_hook(tensorboard_writer: SummaryWriter, mode: str, epoch: int,
                            losses: torch.Tensor, model_outputs: torch.Tensor, targets: torch.Tensor):
        """
        See cvims package hook parameters for NetworkOperate.operate
        """
        if loss_type == 'classification':
            accuracy = calculate_accuracy(
                model_outputs=model_outputs, targets=targets, calc_mean=True
            )
        else:
            model_outputs = torch.round(model_outputs).to(dtype=torch.uint8).squeeze()
            accuracy = (model_outputs == targets).to(dtype=torch.float32).mean().detach()

        tensorboard_writer.add_scalar('/'.join(['Accuracy', mode]), accuracy, epoch)
    
    return {
        'train_after_batch_hook': after_train_batch_hook,
        'eval_after_batch_hook': after_val_batch_hook,
        'after_epoch_hook': after_epoch_hook,
        'tensorboard_hook': tensorboard_hook
    }


def create_dataset(
    data_dir: str,
    data_year: str or int,
    annotation_file_name: str,
    mode: str,  # train or val
    camera_name: str = None,
    specific_labels: List[int] = None,
    include_label_borders: bool = False
):
    assert mode == 'train' or mode == 'val'

    return FusariumDataset(
        data_dir=data_dir,
        data_year=data_year,
        camera_name=camera_name,
        annotation_file_name=annotation_file_name,
        specific_labels=specific_labels,
        include_label_borders=include_label_borders,
        transform=train_transforms(data_year) if mode == 'train' else val_transforms(data_year)
    )


def create_dataset_merge(
    data_dir: str,
    data_year: str or int,
    camera_name: str,
    annotation_file_name: str,
    mode: str,  # train or val
    specific_labels: List[int] = None,
    include_label_borders: bool = False,
    round_targets: bool = False,
    floating_point_labels: bool = False,
    merge_option: str = 'union'  # 'union' or 'intersect'
):
    assert mode == 'train' or mode == 'val'

    annotation_file_names = annotation_file_name.split('+')
    
    if len(annotation_file_names) > 1:
        # multiple years?
        if data_year:
            data_year = data_year.split('+')
            if len(data_year) == 1:
                data_year = [data_year] * len(annotation_file_names)
        
        # multiple cameras?
        if camera_name:
            camera_name = camera_name.split('+')
            if len(camera_name) == 1:
                camera_name = [camera_name] * len(annotation_file_names)

        datasets = []
        for i in range(0, len(annotation_file_names)):
            datasets.append(create_dataset(
                data_dir=data_dir,
                data_year=data_year[i],
                camera_name=camera_name[i] if camera_name else None,
                annotation_file_name=annotation_file_names[i],
                specific_labels=None,
                mode=mode
            ))
        
        ##########################
        # Fuse training datasets #
        ##########################
        return FusariumDatasetMerger(
            fusarium_datasets=datasets,
            specific_labels=specific_labels,
            transform=train_transforms(data_year) if mode == 'train' else val_transforms(data_year),
            label_mode='mean',
            keep_mode=merge_option,
            round_labels=round_targets,
            floating_point_labels=floating_point_labels,
            include_label_borders=include_label_borders
        )
    else:
        return create_dataset(
            data_dir=data_dir,
            data_year=data_year,
            camera_name=camera_name,
            annotation_file_name=annotation_file_names[0],
            specific_labels=specific_labels,
            include_label_borders=include_label_borders,
            mode=mode
        )


def run_model(
    save_path: str,
    configuration: Dict,
    data_dir: str,
    data_year: int or str or List,
    train_annotation_file_name: str or List,
    val_annotation_file_name: str or List,
    test_annotation_file_name: str or List,
    loss_type: str,
    camera_name: str or List = None,
    specific_labels: List[int] = None,
    include_label_borders: bool = False,
    train_dataset: _FusariumDataset = None,
    val_dataset: _FusariumDataset = None,
    round_targets: bool = False,
    floating_point_labels: bool = False,
    merge_option: str = 'union'
) -> Tuple[NetworkOperator, torch.nn.Module]:
    """
    :param save_path: Log directory
    :param configuration: Python dictionary configuration
    :param data_dir: Path to search images based on the annotation file path (excluding the year - see next parameters)
    :param data_year: Year of dataset (sub-folder name). Type None to use all. And pass a list to create a merged multi-year dataset.
    :param train_annotation_file_name: Ttraining annotation file name
    :param val_annotation_file_name: Validation annotation file name
    :param test_annotation_file_name: Test annotation file name
        (does not get loaded for execution, we just put the information into the model configurations
        so that we can evaluate on the correct test_annotation file later.)
    :param loss_type: 'classification' or 'regression'
    :param camera_name: Load specific camera from specific data year of dataset.
    :param specific_labels: Only for loading specific labels of the Fusarium dataset.
    :param include_label_borders: Dataset setting. Only active if specific labels are passed.
    :param train_dataset: Overwrites the initialization of the train dataset by an passed instantiated dataset
    :param val_dataset: Overwrites the initialization of the train dataset by an passed instantiated dataset
    :param round_targets: Only active if two or more datasets are provided. Rounds the targets if multiple datasets have a label for the same image.
    :param floating_point_labels: Allows floating point labels after merging.
    :param merge_option: Union or intersection
    """

    assert loss_type == 'regression' or loss_type == 'classification'
    
    model_name = configuration['model_name']
    save_path = os.path.join(save_path, loss_type, model_name)

    set_seed(42)

    ###########
    # Dataset #
    ###########
    data_loader_settings = configuration['data_loader_settings']

    specific_labels = train_dataset.specific_labels if train_dataset else specific_labels

    train_dataset = train_dataset if isinstance(train_dataset, _FusariumDataset) else create_dataset_merge(
        data_dir=data_dir, data_year=data_year, camera_name=camera_name, annotation_file_name=train_annotation_file_name,
        mode='train', specific_labels=specific_labels, include_label_borders=include_label_borders,
        round_targets=round_targets, floating_point_labels=floating_point_labels, merge_option=merge_option
    )

    train_data_loader = initialize_data_loader(
        batch_size=configuration['batch_size'],
        dataset=train_dataset,
        data_loader_settings=data_loader_settings,
        data_sampler=configuration['data_train_sampler'],
        mode='training'
    )

    val_dataset = val_dataset if isinstance(val_dataset, _FusariumDataset) else create_dataset_merge(
        data_dir=data_dir, data_year=data_year, camera_name=camera_name, annotation_file_name=val_annotation_file_name,
        mode='val', specific_labels=specific_labels, include_label_borders=include_label_borders,
        round_targets=round_targets, floating_point_labels=floating_point_labels, merge_option=merge_option
    )

    val_data_loader = initialize_data_loader(
        batch_size=configuration['batch_size'],
        dataset=val_dataset,
        data_loader_settings=data_loader_settings,
        data_sampler=configuration['data_val_sampler'],
        mode='validation'
    )

    output_classes = train_dataset.unique_labels()
    output_classes_val = val_dataset.unique_labels()

    if output_classes != output_classes_val:
        print('Different output classes between training dataset and validation dataset.')
    
    output_classes = len(output_classes) if loss_type == 'classification' else 1
    
    ##################
    # Early stopping #
    ##################
    early_stopping = None
    if 'early_stopping' in configuration and configuration['early_stopping']:
        early_stopping = EarlyStoppingByLoss(
            patience=10,
            delta=0.001
        )
    
    loss_name = None
    loss_fn = None
    if loss_type == 'classification':
        loss_name = 'cross_entropy'
        # if FusariumMerger is put in as dataset then we have to round the targets first
        def loss_fn(model_outputs, targets):
            return F.cross_entropy(model_outputs, targets)
    else:
        loss_name = 'mse_loss'
        def loss_fn(model_outputs, targets):
            # targets = targets.to(dtype=torch.float32)
            # return torch.sqrt(F.mse_loss(model_outputs.squeeze(dim=1), targets))
            # return F.mse_loss(model_outputs.squeeze(dim=1), targets)
            return F.l1_loss(model_outputs.squeeze(dim=1), targets)
    
    ################
    # Create model #
    ################
    model = load_model(
        name=configuration['model_name'],
        output_classes=output_classes,
        config=configuration['model_parameters']
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    print(get_model_trainable_parameter_count(model=model))

    # Adapt the log dir path if specific labels are set otherwise add 'all_classes' flag

    if data_year:
        save_path = os.path.join(save_path, str(data_year))
        if camera_name:
            save_path = os.path.join(save_path, camera_name)
        else:
            save_path = os.path.join(save_path, 'all_cameras')
    else:
        save_path = os.path.join(save_path, 'all_years')

    if specific_labels:
        save_path = os.path.join(save_path, '_'.join([str(c) for c in sorted(specific_labels)]))

        if include_label_borders:
            save_path = '_'.join([save_path, 'with_label_borders'])
    else:
        save_path = os.path.join(save_path, 'all_classes')

    network_operator = NetworkOperator(model_name=configuration['model_name'], model=model, log_dir=save_path)

    ###################
    # Load parameters #
    ###################
    epochs = configuration['epochs']
    learning_rate = configuration['learning_rate']
    optimizer = configuration['optimizer']
    optimizer = optimizer(model.parameters(), lr=learning_rate, **configuration['optimizer_kwargs'])
    lr_scheduler = configuration['lr_scheduler']
    if lr_scheduler:
        lr_scheduler = lr_scheduler(optimizer=optimizer, **configuration['lr_scheduler_kwargs'])    
    
    #######################
    # Save configurations #
    #######################

    # Add function parameters to configuration before save to disk (for eval purposes)
    information = dict(
        function_parameters=dict(
            save_path=save_path,
            data_dir=data_dir,
            data_year=data_year,
            camera_name=camera_name,
            train_annotation_file_name=train_annotation_file_name,
            val_annotation_file_name=val_annotation_file_name,
            test_annotation_file_name=test_annotation_file_name,
            loss_type=loss_type,
            loss_name=loss_name,
            specific_labels=specific_labels,
            include_label_borders=include_label_borders,
            round_targets=round_targets,
            floating_point_labels=floating_point_labels,
            merge_option=merge_option
        ),
        output_classes=output_classes,
        **configuration
    )

    save_model_information(
        log_dir=network_operator.log_dir,
        file_name='model_configuration',
        information=information
    )

    #################
    # Execute model #
    #################
    best_model = network_operator.operate(
        epochs=epochs, optimizer=optimizer, loss_fn=loss_fn, device=device,
        train_data_loader=train_data_loader, eval_data_loader=val_data_loader,
        **create_hooks(
            train_data_loader=train_data_loader,
            val_data_loader=val_data_loader,
            lr_scheduler=lr_scheduler,
            loss_type=loss_type,
            unfreeze_epoch=configuration['unfreeze_epoch'] if 'unfreeze_epoch' in configuration else 0,
            model=model
        ),
        early_stopping=early_stopping,
        log_active=True,
        use_tensorboard=True,
        save_best_only=True,
        loss_optimization='minimize'
    )

    return network_operator, best_model    


if __name__ == '__main__':
    ###################
    # RUN WITH CONFIG #
    ###################
    config = efficientnet_configuration

    save_path = './runs/'
    
    expert_name = 'rater1'

    assert expert_name, 'Please specify expert name!'

    data_year = 2020

    # camera_name = 'PANA_210706'
    camera_name = None

    if expert_name:
        save_path = os.path.join(save_path, expert_name)

    # if is_expert_annotation (below) is set to True then the data dir is adapted automatically to thie correct path
    data_dir = r'/path/to/data'
    # data_dir = r'/data/departments/schoen/roessle/FHB_classification_public/expert_annotations/rater1'
    train_annotation_file_name='annotations_train_stratified.txt'
    val_annotation_file_name='annotations_val_stratified.txt'
    test_annotation_file_name='annotations_test_stratified.txt'  # only used for configuration and not for training and evalation!

    # add expert name to the beginning of the annotation files names train, val and test
    train_annotation_file_name = '_'.join([expert_name, train_annotation_file_name])
    val_annotation_file_name = '_'.join([expert_name, val_annotation_file_name])
    test_annotation_file_name = '_'.join([expert_name, test_annotation_file_name])


    ###################
    # RUN WITH KWARGS #
    ###################
    # since we usually dont have enough data for all classes we can specify the classes we want to train on
    specific_labels = [3,4,5,6]
    # meaning it includes the cut classes as well
    # e.g. cutting 1,2 gives another combined class (2) containing all data of 1,2 (before the 3, 4, etc.)
    include_label_borders = True

    kwargs = dict(
        data_dir=data_dir,
        specific_labels=specific_labels,
        include_label_borders=include_label_borders,
        save_path=save_path,
        train_annotation_file_name=train_annotation_file_name,
        val_annotation_file_name=val_annotation_file_name,
        test_annotation_file_name=test_annotation_file_name
    )

    #########################
    # start with regression #
    #########################
    loss_type = 'regression'

    assert loss_type == 'regression' or loss_type == 'classification'

    run_model(
            configuration=config,
            data_year=data_year,
            camera_name=camera_name,
            loss_type=loss_type,
            **kwargs
        )

