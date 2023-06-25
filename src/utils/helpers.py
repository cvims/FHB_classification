#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Organization  : Technische Hochschule Ingolstadt
# Institution   : Computer Vision for Intelligent Mobility Systems (CVIMS)
# Created By    : Dominik Roessle
# Created Date  : Fri October 28 2022
# Latest Update : Fri October 28 2022
# =============================================================================
"""
Helper functions
"""
# =============================================================================
# Imports
# =============================================================================
import abc
import os
import copy
from typing import Dict
import torch
from torch.nn import Parameter
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Any, Iterable, List, Tuple, Callable, Iterator
import time
import math
import numpy as np
import random
from torch.utils.data import Sampler, Dataset
from src.utils.dataset import FusariumDataset
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import json



def create_data_sampler(sampler_name: str, dataset: FusariumDataset):
    """
    Creates the data sampler
    """
    if sampler_name == 'weighted_random':
        return WeightedRandomSampler(
            num_samples=dataset.__len__(),
            weights=weight_class_labels(labels=dataset._labels, sum_to_one=False),
            replacement=True
        )
    elif sampler_name == 'undersampling':
        return UndersamplingSampler(
            labels=dataset._labels
        )
    else:
        raise NotImplementedError(f'No sampler implementation for name {sampler_name} available.')


def create_data_loader(dataset: Dataset, workers: int, batch_size: int, **kwargs) -> DataLoader:
    """
    Function to create a train and test data loader.
    """
    assert workers >= 1 and batch_size >= 1
    workers = min(os.cpu_count(), workers)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, **kwargs)

    return data_loader


def initialize_data_loader(batch_size: int, dataset: FusariumDataset, mode: str, data_sampler: str = None, data_loader_settings: Dict = None) -> DataLoader:
    """
    Creates the data loader with the given configuration
    :param batch_size: Batch size
    :param dataset: FusariumDataset
    :param data_loader_settings: Data loader settings from configuration
    :param data_sampler: Dataset sampler
    :param mode: 'training' or 'validation' or 'test' for iteration description (tqdm description).
    """
    if data_sampler:
        data_sampler = create_data_sampler(sampler_name=data_sampler, dataset=dataset)
    
    if not data_loader_settings:
        data_loader_settings = dict(
            workers=1
        )
    
    shuffle = False
    if 'shuffle' in data_loader_settings:
        shuffle = data_loader_settings['shuffle']
        del data_loader_settings['shuffle']

    data_loader = BeautifyDataLoaderIterations(
        data_loader=create_data_loader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False if data_sampler else shuffle,
            sampler=data_sampler,
            **data_loader_settings
        ),
        tqdm_description=f'Iterating {mode} dataset'
    )

    return data_loader


def set_seed(seed: int):
    # set global seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # usually not used inside our scripts
    np.random.seed(seed)
    random.seed(seed)


class BeautifyDataLoaderIterations(Iterable):
    def __init__(self, data_loader: DataLoader, tqdm_description: str = None, tqdm_unit: str = 'batch'):
        """
        Uses tqdm visualizations for dataset iterations.
        :param data_loader: torch data loader
        :param tqdm_description: Visualization text
        :param tqdm_unit: tqdm unit description
        """
        self.data_loader = data_loader
        self._org_next_data = None
        self.tqdm_description = tqdm_description
        self.tqdm_unit = tqdm_unit
        self.tqdm_num = 0
        self.tqdm_bar = None

    def _beautify(self):
        self.tqdm_bar = tqdm(
            desc=self.tqdm_description, total=self.data_loader.__len__(), unit=self.tqdm_unit
        )

    def next_data_wrap(self, old_next_data: Any):
        def _next_data():
            if self.tqdm_num >= self.data_loader.__len__():
                self.tqdm_num = 0
                self.tqdm_bar.close()
                self.tqdm_bar = None
            else:
                self.tqdm_num += 1
                self.tqdm_bar.update(1)
            
            return old_next_data()
        
        return _next_data

    def __iter__(self):
        # open a new tqdm bar
        self._beautify()
        iter_obj = self.data_loader.__iter__()
        iter_obj._next_data = self.next_data_wrap(iter_obj._next_data)
        return iter_obj


def _prepare_accuracy_tensors(model_outputs: torch.Tensor or List, targets: torch.Tensor or List) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    model_outputs = torch.cat(model_outputs, dim=0) if isinstance(model_outputs, list) else model_outputs
    targets = torch.cat(targets, dim=0) if isinstance(targets, list) else targets
    assert not torch.is_floating_point(targets), 'Please provide an integer data type tensor for targets.'
    assert len(targets.size()) <= 2, 'Please provide a target tensor of max 2-dimensions, ' \
                                     f'got len(targets.size()) dimensions.'
    if len(targets.size()) == 2:
        assert targets.size()[1] == 1, f'Please provide one label per batch, got {targets.size()[2]}.'
    if len(model_outputs.size()) == 1:
        # could be regression output
        if torch.is_floating_point(model_outputs):
            model_outputs = torch.round(model_outputs)
    elif len(model_outputs.size()) == 2:
        if model_outputs.size()[1] == 1:
            # accuracy through regression
            model_outputs = torch.flatten(model_outputs)
            model_outputs = torch.round(model_outputs)
        else:
            # accuracy through classification
            model_outputs = torch.max(model_outputs, dim=1).indices
    else:
        raise AttributeError('Can only process model output tensors of dimension size 1, e.g. [n] or 2, e.g. [n, m], '
                             f'got {model_outputs.size()}')

    model_outputs = model_outputs.to(dtype=targets.dtype, device=targets.device)

    return model_outputs.detach(), targets.detach()


def calculate_accuracy(model_outputs: torch.Tensor or List, targets: torch.Tensor or List, calc_mean: bool = True)\
        -> torch.Tensor:
    """
    Uses models outputs and targets to calculate the accuracy
    :param model_outputs: Model outputs
    :param targets: Target labels
    :param calc_mean: If True the accuracy is one number (as usual) if False then the return is a tensor of
    model output and target comparisons
    :return:
    """
    # accuracy
    # stacking at dim 0 only available if network batch size output is permanently equal
    model_outputs, targets = _prepare_accuracy_tensors(
        model_outputs=model_outputs, targets=targets
    )
    if calc_mean:
        return (model_outputs == targets).to(dtype=torch.float32).mean()
    else:
        return model_outputs == targets


class CustomEarlyStopping(metaclass=abc.ABCMeta):
    def __init__(self, patience: int, delta: float, optimization: str) -> None:
        """
        Early stopping interface as default base for new early stopping mechanisms.
        :param patience: Epochs to wait until firing to stop
        :param delta: Difference threshold to whether use the following epoch as improvement or not
        :param optimization: String value ('minimize' or 'maximize'). Optimization direction.
        """
        assert patience > 0
        self.patience = patience
        assert 0.0 < delta <= 1.0
        self.delta = delta
        assert optimization == 'maximize' or optimization == 'minimize'
        self.optimization = optimization
        self.best_score = None
        self.patience_count = 0

        # Variable to call to get the early stop state
        self.early_stop = False

    def reset_stopping_scores(self):
        """
        Reset previously calculated values for early stopping
        :return:
        """
        self.best_score = None
        self.patience_count = 0
        self.early_stop = False

    @staticmethod
    def _preprocess_calculate_score_inputs(
            losses: torch.Tensor or List[float],
            model_outputs: List[torch.Tensor] or List[float],
            targets: List[torch.Tensor] or List[float]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param losses: Model losses
        :param model_outputs: Outputs of the network
        :param targets: Corresponding targets to predictions
        :return:
        """
        if isinstance(losses, list):
            losses = torch.Tensor(losses)
        if isinstance(model_outputs, list):
            model_outputs = torch.Tensor(model_outputs)
        if isinstance(targets, list):
            targets = torch.Tensor(targets)

        return losses, model_outputs, targets

    @abc.abstractmethod
    def _calculate_score(self, losses: torch.Tensor or List[float],
                         model_outputs: List[torch.Tensor] or List[float],
                         targets: List[torch.Tensor] or List[float]) -> float:
        """
        Method to implement to calculate to early stopping score.
        :param losses: Losses list of tensors
        :param model_outputs:
        :param targets:
        :return:
        """
        raise NotImplementedError

    def __call__(self, losses: torch.Tensor or List[float],
                 model_outputs: List[torch.Tensor] or List[float],
                 targets: List[torch.Tensor] or List[float]) -> None:
        """
        Early stopping routine
        :param losses: Model losses
        :param model_outputs: Outputs of the network
        :param targets: Corresponding targets to predictions
        :return:
        """
        calculated_score = self._calculate_score(
            losses=losses, model_outputs=model_outputs, targets=targets
        )

        if self.best_score is None:
            self.best_score = calculated_score
            return

        if self.optimization == 'maximize':
            if calculated_score > self.best_score + self.delta:
                self.best_score = calculated_score
                self.patience_count = 0
            else:
                self.patience_count += 1
        elif self.optimization == 'minimize':
            if calculated_score < self.best_score - self.delta:
                self.best_score = calculated_score
                self.patience_count = 0
            else:
                self.patience_count += 1

        if self.patience_count > self.patience:
            self.early_stop = True
        else:
            self.early_stop = False


class EarlyStoppingByLoss(CustomEarlyStopping):
    def __init__(self, patience: int, delta: float = 0.0) -> None:
        """
        Early stopping mechanism by using the loss values
        :param patience:
        :param delta:
        """
        super(EarlyStoppingByLoss, self).__init__(patience=patience, delta=delta, optimization='minimize')

    def _calculate_score(self, losses: torch.Tensor or List[float],
                         model_outputs: List[torch.Tensor] or List[float],
                         targets: List[torch.Tensor] or List[float]) -> float:
        """
        :param losses: Model losses
        :param model_outputs: Outputs of the network
        :param targets: Corresponding targets to predictions
        :return:
        """
        losses, model_outputs, targets = self._preprocess_calculate_score_inputs(
            losses=losses, model_outputs=model_outputs, targets=targets
        )

        avg_loss = torch.sum(losses) / torch.flatten(losses).size()[0]
        return avg_loss
    


def save_model(model: torch.nn.Module, log_dir: str, file_name: str, model_state_key: str, **kwargs) -> None:
    """
    Saves the torch model and other information (kwargs) into the log_dir
    :param model: torch model (nn.Module)
    :param log_dir: Directory to save the model. The class defined log dir is used as directory.
    :param model_state_key: Key of dict entry do differentiate the model from kwargs.
    :param file_name: File name without extension.
    :param kwargs: All additional information that should be stored (torch.save)
    :return:
    """
    info_dict = {
        model_state_key: model.state_dict(),
        **kwargs
    }

    # in case a file extension is provided
    file_name = file_name.split('.')[0]

    save_dir = os.path.join(log_dir)
    old_mask = os.umask(000)
    os.makedirs(save_dir, exist_ok=True)

    # save the information
    torch.save(info_dict, os.path.join(save_dir, file_name + '.pth'))

    os.umask(old_mask)


def build_timestamp_log_dir(log_dir: str) -> str:
    """
    Creates a default log dir based on timestamp
    :param log_dir: Base directory
    :return:
    """
    # format: day month year hour minute second
    time_obj = time.localtime(time.time())
    timestamp = '%d%d%d_%d%d%d' % (time_obj.tm_mday, time_obj.tm_mon, time_obj.tm_year,
                                   time_obj.tm_hour, time_obj.tm_min, time_obj.tm_sec)

    return os.path.join(log_dir, timestamp)


def load_model(model: torch.nn.Module, path: str, to_device: torch.device, model_state_key: str = None)\
        -> Tuple[torch.nn.Module, Any]:
    """
    Loads the torch model and other additional information.
    :param model: torch Module where to load the state dict into.
    :param path: full path (with filename) to the models state dictionary.
    :param to_device: torch.device to put the model
    :param model_state_key: Key of models load_state_dict to find the model specific weights
    :return:
    """
    assert os.path.isfile(path=path)

    model_info = torch.load(path, map_location=to_device)
    if model_state_key:
        model.load_state_dict(model_info[model_state_key])
    else:
        model.load_state_dict(model_info)

    model.eval()
    model = model.to(device=to_device)

    return model, model_info


def set_data_device(data: Dict[str, torch.Tensor] or List[torch.Tensor] or torch.Tensor or None,
                    labels: Dict[str, torch.Tensor] or List[torch.Tensor] or torch.Tensor or None,
                    device: torch.device) \
        -> Tuple[Dict[str, torch.Tensor] or List[torch.Tensor] or torch.Tensor,
                 Dict[str, torch.Tensor] or List[torch.Tensor] or torch.Tensor]:
    """
    Sets the data to the torch device of your choice for different formats (dict, list, or tensors)
    :param data: Data to set the device
    :param labels: Corresponding labels to set the device
    :param device: torch.device
    :return: data, label of same types as the input parameters
    """
    # set device for data
    if data is not None:
        if isinstance(data, dict):
            # iterate all data elements
            for data_name in data:
                if data[data_name] is not None:
                    data[data_name] = data[data_name].to(device=device)
        elif isinstance(data, list):
            for i, d in enumerate(data):
                data[i] = d.to(device=device)
        else:
            data = data.to(device=device)

    # set device for labels
    if labels is not None:
        tensorable = True
        if isinstance(labels, Iterable):
            for i, l in enumerate(labels):
                # todo tuple handling
                if isinstance(l, dict):
                    tensorable = False
                    for l_name in l:
                        l[l_name] = l[l_name].to(device=device)
                else:
                    # torch.Tensor
                    labels[i] = l.to(device=device)
        
        if tensorable:
            # torch.Tensor
            labels = labels.to(device=device)
    
    if isinstance(labels, List) or isinstance(labels, Tuple):
        labels = torch.tensor(labels)

    return data, labels
    

class NetworkOperator(abc.ABC):
    def __init__(self, model: torch.nn.Module, model_name: str, log_dir: str or None) -> None:
        """
        Creates a network operator with predefined logging and operating functionality.
        :param model: torch model
        :param model_name: Representative torch model name
        :param log_dir: Logging directory
        """
        assert model is not None and isinstance(model, torch.nn.Module)
        self.model = model
        self.model_name = model_name

        self._state_dict_name = 'model_state_dict'

        # Once initialized it is fixed
        self._log_dir = None
        self.log_dir = log_dir

    @property
    def log_dir(self) -> str:
        """
        Global network operator log directory getter
        :return:
        """
        return self._log_dir

    @log_dir.setter
    def log_dir(self, path: str):
        """
        Global network operator log directory setter
        :param path: Log directory path
        :return:
        """
        if self._log_dir:
            # cannot be reinitialized
            return

        if path:
            self._log_dir = self._create_log_dir_path(log_dir=path)

    def overwrite_log_dir(self, path: str):
        """
        Only opportunity to overwrite a defined log dir
        :param path:
        :return:
        """
        self._log_dir = self._create_log_dir_path(log_dir=path)

    @staticmethod
    def _create_log_dir_path(log_dir: str) -> str:
        """
        Creates the path for the log directory
        :param log_dir: Base directory
        :return:
        """
        return build_timestamp_log_dir(log_dir=log_dir)

    def save_model(self, file_name: str, **kwargs) -> None:
        """
        Saves the torch model and other information (kwargs) into the log_dir
        :param file_name: File name without extension.
        :param kwargs: All additional information that should be stored (torch.save)
        :return:
        """
        assert self.log_dir, 'Provide a log dir for the network operator before starting the operation'

        save_model(
            model=self.model, log_dir=self.log_dir, file_name=file_name, model_state_key=self._state_dict_name, **kwargs
        )

    def load_model(self, file_name: str, to_device: torch.device)\
            -> Tuple[torch.nn.Module, Any]:
        """
        Loads the torch model and other additional information.
        :param file_name: File name without extension.
        :param to_device: torch.device to put the model
        :return:
        """
        assert self.log_dir, 'Provide a log dir for the network operator before starting the operation'

        return load_model(
            model=self.model, path=os.path.join(self.log_dir, file_name), model_state_key=self._state_dict_name,
            to_device=to_device
        )

    def _model_iteration(
            self, device: torch.device, data_loader: Iterable, optimizer: Optimizer = None,
            loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
            model_input_pre_hook: Callable[[Any, Any], Tuple[Any, Any]] = None,
            optimizer_pre_hook: Callable[[Iterator[Parameter]], Any] = None,
            after_batch_hook: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, ...], Any] = None,
            after_iterations_hook: Callable[..., Any] = None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Model iteration implementation with different hook interactions.
        :param device: torch.device
        :param data_loader: Iterable object used for data looping. The first two objects from a iteration must be data
        and label where data is a torch.Tensor and label is the corresponding annotation (also torch.Tensor).
        :param optimizer: torch optimizer
        :param loss_fn: Loss function
        :param model_input_pre_hook: Hook to interact with data and label, respectively before inputting it to the
        network
        :param optimizer_pre_hook: Hook for analysing and updating the model parameters. Input gets parameters which are
        the model parameters. The function output is ignored.
        :param after_batch_hook: Hook for every batch iteration of the data loader. Gets called at the end of each
        iteration with inputs (losses, model outputs, targets, *args) where args are the outputs from the data loader
        :param after_iterations_hook: Hook at the end of the iteration. For example learning schedulers.
        :param kwargs: Kwargs for the model forward pass.
        :return:
        """
        losses, full_losses, model_outputs, targets = [], [], [], []

        for data in data_loader:
            batch, labels, *other_data = data
            # set the device of the batch and labels
            batch, labels = set_data_device(data=batch, labels=labels, device=device)

            if model_input_pre_hook:
                batch, labels = model_input_pre_hook(batch, labels)

            # call the model forward
            model_output = self.model.forward(batch, **kwargs)

            _loss = None
            _full_loss = None
            if loss_fn:
                # Calculate loss
                _loss = _full_loss = loss_fn(model_output, labels)
        
            # if _loss is not a tensor then the first element of the Iterable object must be the tensor (backwards)
            if isinstance(_loss, Iterable) and not isinstance(_loss, torch.Tensor):
                _loss = _loss[0]

            if optimizer:
                # Update models parameters
                optimizer.zero_grad()

                # we only go backwards if the optimizer is set
                _loss.backward()

                # Call pre optimizer hook if there is one defined
                if optimizer_pre_hook:
                    optimizer_pre_hook(self.model.parameters())

                optimizer.step()

            # Collect targets and outputs and loss
            if _loss:
                losses.append(_loss.detach())
                if _loss != _full_loss:
                    full_losses.append(_full_loss)
            _targets = labels.detach() if isinstance(labels, torch.Tensor) else labels
            targets.append(_targets)

            _model_outputs = model_output.detach() if isinstance(model_output, torch.Tensor) else model_output
            model_outputs.append(_model_outputs)

            # Call after batch hook
            if after_batch_hook:
                after_batch_hook(_full_loss, _targets, _model_outputs, *data)

        # make tensors out of losses, model outputs and targets
        losses = torch.stack(losses) if losses else None
        if len(model_outputs) > 0 and isinstance(model_outputs[0], torch.Tensor):
            model_outputs = torch.cat(model_outputs, dim=0)
        if len(targets) > 0 and isinstance(targets[0], torch.Tensor):
            targets = torch.cat(targets, dim=0)

        # Call after epoch hook
        if after_iterations_hook:
            after_iterations_hook()
        
        ret_loss = losses
        if full_losses:
            ret_loss = (losses, full_losses)

        return ret_loss, model_outputs, targets

    def train_epoch(
            self, device: torch.device, data_loader: Iterable, optimizer: Optimizer,
            loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            model_input_pre_hook: Callable[[Any, Any], Tuple[Any, Any]] = None,
            optimizer_pre_hook: Callable[[Iterator[Parameter]], Any] = None,
            after_batch_hook: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, ...], Any] = None,
            after_iteration_hook: Callable[..., Any] = None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Train epoch implementation with different hook interactions.
        :param device: torch.device
        :param data_loader: Iterable object used for data looping. The first two objects from a iteration must be data
        and label where data is a torch.Tensor and label is the corresponding annotation (also torch.Tensor).
        :param optimizer: torch optimizer
        :param loss_fn: Loss function
        :param model_input_pre_hook: Hook to interact with data and label, respectively before inputting it to the
        network
        :param optimizer_pre_hook: Hook for analysing and updating the model parameters. Input gets parameters which are
        the model parameters. The function output is ignored.
        :param after_batch_hook: Hook for every batch iteration of the data loader. Gets called at the end of each
        iteration with inputs (losses, model outputs, targets, *args) where args are the outputs from the data loader
        :param after_iteration_hook: Hook at the end of the iteration. For example learning schedulers.
        :param kwargs: Kwargs for the model forward pass.
        :return:
        """
        # set training mode for model
        self.model.train()

        with torch.enable_grad():
            return self._model_iteration(
                device=device, data_loader=data_loader, optimizer=optimizer,
                model_input_pre_hook=model_input_pre_hook, optimizer_pre_hook=optimizer_pre_hook,
                loss_fn=loss_fn, after_batch_hook=after_batch_hook, after_iterations_hook=after_iteration_hook, **kwargs
            )

    def eval_iter(
            self, device: torch.device, data_loader: Iterable,
            loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
            model_input_pre_hook: Callable[[Any, Any], Tuple[Any, Any]] = None,
            after_batch_hook: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Any] = None,
            **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Eval epoch implementation with different hook interactions.
        :param device: torch.device
        :param data_loader: Iterable object used for data looping. The first two objects from a iteration must be data
        and label where data is a torch.Tensor and label is the corresponding annotation (also torch.Tensor).
        :param loss_fn: Loss function
        :param model_input_pre_hook: Hook to interact with data and label, respectively before inputting it to the
        network
        :param after_batch_hook: Hook for every batch iteration of the data loader. Gets called at the end of each
        iteration with inputs (losses, model outputs, targets, *args) where args are the outputs from the data loader
        :param kwargs: Kwargs for the model forward pass.
        :return:
        """
        # set eval mode for model
        self.model.eval()

        with torch.no_grad():
            return self._model_iteration(
                device=device, data_loader=data_loader, optimizer=None,
                model_input_pre_hook=model_input_pre_hook, optimizer_pre_hook=None,
                loss_fn=loss_fn, after_batch_hook=after_batch_hook, after_iterations_hook=None, **kwargs
            )

    def operate(
            self, epochs: int, device: torch.device, train_data_loader: Iterable, eval_data_loader: Iterable,
            optimizer: Optimizer, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            model_input_pre_hook: Callable[[Any, Any], Tuple[Any, Any]] = None,
            optimizer_pre_hook: Callable[[Iterator[Parameter]], None] = None,
            train_after_batch_hook: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None] = None,
            eval_after_batch_hook: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None] = None,
            after_epoch_hook: Callable[Tuple[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                                        ], None] = None,
            early_stopping: CustomEarlyStopping = None,
            log_active: bool = True, use_tensorboard: bool = False, save_best_only: bool = False,
            tensorboard_hook: Callable[[SummaryWriter, str, int, torch.Tensor, torch.Tensor, torch.Tensor],
                                       None] = None,
            loss_optimization: str = 'minimize', **kwargs
    ) -> torch.nn.Module:
        """
        Combines train and eval and adds logging mechanisms and operations to it. This is a wrapper method for training.
        :param epochs: Epochs for training and evaluation.
        :param device: torch device
        :param train_data_loader: data loader for training iterations
        :param eval_data_loader: data loader for evaluation iterations
        :param optimizer: torch optimizer
        :param loss_fn: Loss function
        :param model_input_pre_hook: Hook to interact with data and label, respectively before inputting it to the
        network
        :param optimizer_pre_hook: Hook for analysing and updating the model parameters. Input gets parameters which are
            the model parameters. The function output is ignored.
        :param train_after_batch_hook: Hook for every batch iteration of the training data loader. Gets called at the
            end of each  iteration with inputs (losses, model outputs, targets, *args) where args are the outputs from
            the data loader
        :param eval_after_batch_hook: Hook for every batch iteration of the training data loader. Gets called at the
            end of each  iteration with inputs (losses, model outputs, targets, *args) where args are the outputs from
            the data loader
        :param after_epoch_hook: Hook at the end of every epoch, after train and eval execution.
            Contains the epoch no (first param) and two tuples as inputs which contain losses, model outputs and targets
            for train and eval, respectively.
        :param early_stopping: Early stopping (CustomEarlyStopping implementation)
        :param log_active: If False then 'use_tensorboard' and 'save_best_only' and all logging events are ignored
        :param use_tensorboard: Opens a tensorboard writer if true else no logging with tensorboard is applied.
        :param save_best_only: Saves only the best model if True else every epoch.
        :param tensorboard_hook: Callable for additionally writing into the tensorboard log.
            Parameters are (tensorboard_writer: SummaryWriter, mode: str, epoch: int, losses: tensor,
            model_outputs: tensor, targets: tensor).
            Parameter 'mode' is a string which contains either train (for training information) or eval (for evaluation
            information). Use tensorboard_writer object to add information to the tensorboard.
        :param loss_optimization: Optimization direction for deciding for best model. Possible values are 'minimize' or
            'maximize'
        :param kwargs: Kwargs for the model forward pass.
        :return: Best model (torch.nn.Module)
        """
        best_model = None
        loss_optimization = loss_optimization.lower()
        assert loss_optimization == 'minimize' or loss_optimization == 'maximize'

        if log_active:
            assert self.log_dir, 'Provide a log dir for the network operator before starting the operation'

        tensorboard_summary_writer = None
        if log_active and use_tensorboard:
            tensorboard_summary_writer = SummaryWriter(log_dir=self.log_dir)

        # for keeping track of best model
        prev_best_loss = math.inf if loss_optimization == 'minimize' else -math.inf

        for epoch in range(0, epochs):
            print('Epoch:', epoch)

            # model training
            t_losses, t_model_outputs, t_targets = self.train_epoch(
                device=device, data_loader=train_data_loader, optimizer=optimizer,
                optimizer_pre_hook=optimizer_pre_hook, loss_fn=loss_fn,
                model_input_pre_hook=model_input_pre_hook,
                after_batch_hook=train_after_batch_hook, after_iteration_hook=None, **kwargs
            )

            # model evaluation
            e_losses, e_model_outputs, e_targets = self.eval_iter(
                device=device, data_loader=eval_data_loader, loss_fn=loss_fn,
                model_input_pre_hook=model_input_pre_hook,
                after_batch_hook=eval_after_batch_hook, **kwargs
            )

            # losses
            full_t_losses = t_losses
            full_e_losses = e_losses
            if isinstance(t_losses, Iterable):
                t_losses = t_losses[0]
                e_losses = e_losses[0]
                t_avg_loss = torch.mean(full_t_losses[0])
                e_avg_loss = torch.mean(full_e_losses[0])
            else:
                t_avg_loss = torch.mean(t_losses)
                e_avg_loss = torch.mean(e_losses)

            # tensorboard interaction
            if use_tensorboard and tensorboard_summary_writer:
                # add known train and eval information
                tensorboard_summary_writer.add_scalar('Loss/train', t_avg_loss, epoch)
                tensorboard_summary_writer.add_scalar('Loss/val', e_avg_loss, epoch)

                # check if tensorboard hook exists
                if tensorboard_hook:
                    # for train information
                    tensorboard_hook(tensorboard_summary_writer, 'train', epoch, full_t_losses, t_model_outputs, t_targets)
                    # for eval information
                    tensorboard_hook(tensorboard_summary_writer, 'eval', epoch, full_e_losses, e_model_outputs, e_targets)

            # save best model
            if (loss_optimization == 'minimize' and prev_best_loss > e_avg_loss) or \
                    (loss_optimization == 'maximize' and prev_best_loss < e_avg_loss):
                prev_best_loss = e_avg_loss
                best_model = copy.deepcopy(self.model)

            if log_active:
                self.save_model(
                    file_name='best_model.pth'.format(epoch=epoch)
                )

                # save for each epoch too
                if not save_best_only:
                    self.save_model(
                        file_name='model_epoch_{epoch}.pth'.format(epoch=epoch)
                    )

            # call the hook if set
            if after_epoch_hook:
                after_epoch_hook(
                    epoch, (full_t_losses, t_model_outputs, t_targets), (full_e_losses, e_model_outputs, e_targets)
                )

            # early stopping
            if early_stopping:
                early_stopping(losses=e_losses, model_outputs=e_model_outputs, targets=e_targets)

                if early_stopping.early_stop:
                    print(f'Early stopping at epoch {epoch}')
                    break

        # close the tensorboard writer
        if tensorboard_summary_writer:
            tensorboard_summary_writer.close()

        return best_model


def get_model_trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model_information(log_dir: str, file_name: str, information: Dict[str, Any]) -> None:
    """
    Saves the torch model and other information (kwargs) into the log_dir
    :param log_dir: Directory to save the model. The class defined log dir is used as directory.
    :param file_name: File name without extension.
    :param information: All additional information that should be stored (torch.save)
    :return:
    """

    # in case a file extension is provided
    file_name = file_name.split('.')[0]

    save_dir = os.path.join(log_dir)
    old_mask = os.umask(000)
    os.makedirs(save_dir, exist_ok=True)

    # save the information
    torch.save(information, os.path.join(save_dir, file_name + '.pth_info'))

    os.umask(old_mask)


def weight_class_labels(labels: List[int], sum_to_one: bool = False):
    """
    Weights the labels by their id to get an equal data sampling of labels when using for example the WeightedRandomSampler.
    """
    label_counts = {label: 0 for label in set(labels)}
    for label in labels:
        label_counts[label] += 1

    len_labels = len(labels)
    weights = np.array([len_labels / label_counts[label] for label in labels])
    if sum_to_one:
        weights /= sum(weights)

    return weights


class UndersamplingSampler(Sampler):
    """
    Samples elements from [0,..,len(weights)-1] with given probabilities (weights).

    Arguments:
        labels (sequence)   : a sequence of all labels
    """

    def __init__(self, labels: List):
        super(UndersamplingSampler, self).__init__(data_source=labels)
        if not isinstance(labels, List) or not labels:
            raise ValueError("labels should be a non-empty list "
                             "value, but got samples={}".format(labels))

        self.labels = labels
        # remove random samples from over labeled classes
        self.undersampled_labels = self._undersample(labels=labels)

    @staticmethod
    def _undersample(labels: List):
        label_indices = {}
        for i, label in enumerate(labels):
            if label not in label_indices:
                label_indices[label] = []

            label_indices[label].append(i)

        label_indices = {label: torch.as_tensor(label_indices[label]) for label in label_indices}

        # calc random indices to delete
        label_counts = {label: label_indices[label].size()[0] for label in label_indices}
        min_label_count = min([label_counts[label] for label in label_counts])

        labels_indices = torch.cat(
            [label_indices[label][torch.randperm(label_counts[label])[:min_label_count]] for label in label_indices])
        labels_indices = labels_indices[torch.randperm(n=labels_indices.size()[0])]

        return labels_indices

    def __iter__(self):
        return iter(self.undersampled_labels)

    def __len__(self):
        return len(self.undersampled_labels)


class WeightedRandomSampler(Sampler):
    """
    Samples elements from [0,..,len(weights)-1] with given probabilities (weights).

    Arguments:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
    """
    def __init__(self, weights, num_samples, replacement=True):
        super(WeightedRandomSampler, self).__init__(data_source=None)
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples


def confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor, output_path: str,
                     id_class_name_map: Dict = None, exists_ok: bool = False) -> Tuple[np.array, pd.DataFrame]:
    if not exists_ok and os.path.isfile(output_path):
        raise FileExistsError('Output path {} already exists.'.format(output_path))

    if predictions.size() != targets.size():
        predictions, targets = _prepare_accuracy_tensors(
            model_outputs=predictions, targets=targets
        )

    if predictions.is_floating_point():
        # round tensor (maybe regression task)
        predictions = predictions.round().to(dtype=targets.dtype, device=targets.device)

    predictions = predictions.flatten()

    class_ids = [tensor.item() for tensor in torch.unique(torch.cat((targets, predictions)))]
    if id_class_name_map:
        class_names = [id_class_name_map[class_id] for class_id in class_ids]
    else:
        class_names = class_ids

    # predictions and targets are now aligned
    # calculate values for the confusion matrix
    cf_values = sk_confusion_matrix(y_true=targets.cpu(), y_pred=predictions.cpu())
    # flattened percentages for labeling
    cf_percentages_rows = np.array([value / sum(values) for values in cf_values for value in values])

    labels = np.asarray(['{:.2%}\nSamples: {}'.format(_percentage, _count) for _percentage, _count in
                         zip(cf_percentages_rows, cf_values.flatten())]).reshape(cf_values.shape)

    df_cm = pd.DataFrame(
        cf_percentages_rows.reshape(cf_values.shape) * 10, index=[i for i in class_names],
        columns=[class_name for class_name in class_names]
    )

    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=labels, fmt='')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.savefig(output_path)
    plt.close()

    return cf_values, df_cm


def dict_to_txt(dictionary: Dict, output_file_path: str, store_functions_as_string: bool = True) -> None:
    """
    Transfers the dictionary into a txt file
    :param dictionary: Dictionary to write to file
    :param output_file_path: Full file path
    :return: None
    """

    if store_functions_as_string:
        validate_dict_to_json(dictionary=dictionary)

    with open(output_file_path, 'w') as file:
        json.dump(dictionary, file, indent='\t')


def str_list_to_txt(lines: List[List[Any]], output_file_path: str, header: List[str]) -> None:
    """
    Stores string lines to a txt file
    :param header: Header of file
    :param lines: List of string (txt) lines
    :param output_file_path: Full file path
    :return:
    """
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w') as file:
        if header:
            file.write(' '.join(col for col in header))
        for line in lines:
            file.write('\n' + ' '.join([elem for elem in line]))


def tensors_to_txt(tensors: List[torch.Tensor], output_file_path: str, tensor_names: List[str] = None):
    if not tensor_names:
        tensor_names = ['\n'] * len(tensors)
    lines = []
    for i, (tensor, tensor_name) in enumerate(zip(tensors, tensor_names)):
        if i != 0 or tensor_name != '\n':
            lines.append(tensor_name + '\n')
        lines.append(np.array_str(tensor.cpu().numpy()) + '\n')

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w') as file:
        file.writelines(lines)


def sklearn_confusion_matrix_to_txt(confusion_matrix: np.array, output_file_path: str) -> None:
    """
    Takes the numpy array as input as produces readable files
    :param confusion_matrix:
    :param output_file_path:
    :return:
    """
    # row is actual value
    # col is predicted value
    header = [*[str(i) for i in range(0, len(confusion_matrix))]]
    df = pd.DataFrame(confusion_matrix, columns=header, index=header)
    with open(output_file_path, 'w') as f:
        f.write(df.to_string(header=True, index=True))


def validate_dict_to_json(dictionary: Dict):
    def recursive_finder(node: Any):
        if isinstance(node, Iterable) and not isinstance(node, str):
            if isinstance(node, dict):
                for k in node:
                    node[k] = recursive_finder(node[k])
            elif isinstance(node, tuple) or isinstance(node, set):
                n_t = []
                for elem in node:
                    n_t.append(recursive_finder(elem))
                if isinstance(node, set):
                    node = set(n_t)
                else:
                    node = (*n_t,)
            else:
                for i, elem in enumerate(node):
                    node[i] = recursive_finder(elem)
        else:
            if isinstance(node, Callable):
                return str(node.__name__)
        
        return node

    for k in dictionary:
        dictionary[k] = recursive_finder(dictionary[k])
    
    return dictionary
