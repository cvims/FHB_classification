#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Organization  : Technische Hochschule Ingolstadt
# Institution   : Computer Vision for Intelligent Mobility Systems (CVIMS)
# Created By    : Dominik Roessle
# Created Date  : Thu October 27 2022
# Latest Update : Thu October 27 2022
# =============================================================================
"""
Evaluation script for Fusarium classification
"""
# =============================================================================
# Imports
# =============================================================================
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import shutil
from typing import Dict, List, Tuple
import torch
from sklearn.metrics import f1_score
from src.utils.models import load_model as init_model
from src.utils.dataset import FusariumDataset, FusariumDatasetMerger
from src.utils.transforms import test_transforms
from src.utils.helpers import initialize_data_loader, set_seed, set_data_device, load_model, calculate_accuracy, \
    confusion_matrix, dict_to_txt, sklearn_confusion_matrix_to_txt, tensors_to_txt, str_list_to_txt


def model_evaluation_iterator(run_paths: Dict, device: torch.device):
    for type in run_paths:
        # regression or classification
        model_types = run_paths[type]
        if not model_types:
            continue
        for model_name in model_types:
            for year in model_types[model_name]:
                for cameras in model_types[model_name][year]:
                    for specific_classes in model_types[model_name][year][cameras]:
                        for model_path, config_path in model_types[model_name][year][cameras][specific_classes]:
                            config = torch.load(f=config_path)
                            config_model_name = config['model_name']
                            config_model_parameters = config['model_parameters']
                            model = init_model(
                                name=config_model_name,
                                output_classes=config['output_classes'],
                                config=config_model_parameters
                            )
                            model, _ = load_model(model=model, path=model_path, to_device=device, model_state_key='model_state_dict')

                            yield model, config, type, os.path.dirname(config_path), model_name, year, cameras, specific_classes


def find_run_paths(start_dir: str):
    """
    Searching for training runs
    """
    def resolve_dir(_dir):
        return os.listdir(_dir) if os.path.isdir(_dir) else []
    

    dict_structure = {}

    for name in resolve_dir(start_dir):
        dict_structure[name] = {}
        model_name_path = os.path.join(start_dir, name)
        for year in resolve_dir(model_name_path):
            dict_structure[name][year] = {}
            year_path = os.path.join(model_name_path, year)
            for camera_name in resolve_dir(year_path):
                dict_structure[name][year][camera_name] = {}
                camera_path = os.path.join(year_path, camera_name)
                for specific_classes in resolve_dir(camera_path):
                    runs_path = os.path.join(year_path, camera_name, specific_classes)
                    dict_structure[name][year][camera_name][specific_classes] = []
                    model_runs = os.listdir(runs_path)
                    for run_name in model_runs:
                        exec_path = os.path.join(runs_path, run_name)
                        best_model_path = os.path.join(exec_path, 'best_model.pth')
                        best_model_path = best_model_path if os.path.isfile(best_model_path) else None
                        model_config_path = os.path.join(exec_path, 'model_configuration.pth_info')
                        model_config_path = model_config_path if os.path.isfile(model_config_path) else None

                        if not best_model_path or not model_config_path:
                            continue

                        dict_structure[name][year][camera_name][specific_classes].append((best_model_path, model_config_path))
    
    return dict_structure


def load_run_paths(root_path: str):
    """
    Searching for classification and regression runs
    """
    regression_path = os.path.join(root_path, 'regression')
    classification_path = os.path.join(root_path, 'classification')

    regression_models = find_run_paths(start_dir=regression_path)
    classification_models = find_run_paths(start_dir=classification_path)

    return {
        'regression': regression_models,
        'classification': classification_models
    }


def create_test_dataset(
    data_dir: str,
    data_year: str or int,
    annotation_file_name: str,
    camera_name: str = None,
    specific_labels: List[int] = None,
    include_label_borders: bool = False,
    label_mode: str = 'mean',
    keep_mode: str = 'intersection',
    round_targets: bool = False,
    floating_point_labels: bool = False,
    transforms = None
):
    test_dataset = None

    annotation_file_names = annotation_file_name.split('+')

    if len(annotation_file_names) > 1:
        datasets = []
        for annotation_file_name in annotation_file_names:
            datasets.append(FusariumDataset(
                data_dir=data_dir,
                data_year=data_year,
                camera_name=camera_name,
                annotation_file_name=annotation_file_name,
                # specific_labels=specific_labels,
                transform=transforms if transforms else test_transforms(data_year=data_year)
            ))
        
        test_dataset = FusariumDatasetMerger(
            fusarium_datasets=datasets,
            specific_labels=specific_labels,
            include_label_borders=include_label_borders,
            transform=transforms if transforms else test_transforms(data_year=data_year),
            label_mode=label_mode,
            keep_mode=keep_mode,
            round_labels=round_targets,
            floating_point_labels=floating_point_labels
        )
    else:
        test_dataset = FusariumDataset(
            data_dir=data_dir,
            data_year=data_year,
            camera_name=camera_name,
            annotation_file_name=annotation_file_names[0],
            transform=transforms if transforms else test_transforms(data_year=data_year),
            specific_labels=specific_labels,
            include_label_borders=include_label_borders
        )
    
    return test_dataset


def eval_model_on_test_dataset(
    model: torch.nn.Module,
    data_dir: str,
    data_year: str or int,
    test_annotation_file_name: str,
    specific_classes: str or List[int],
    device: torch.device,
    camera_name: str = None,
    include_label_borders: bool = False,
    # for dataset merger (if multiple test annotation files are to be used - separated by '+')
    label_mode: str = 'mean',
    keep_mode: str = 'intersection',
    floating_point_labels: bool = False,
    undersampling: bool = False,
    transforms = None
):
    
    if isinstance(specific_classes, str):
        specific_classes = [int(label) for label in specific_classes.split('_') if label.isdigit()]

    # create dataset and data loaders
    test_dataset = create_test_dataset(
        data_dir=data_dir,
        data_year=data_year,
        camera_name=camera_name,
        annotation_file_name=test_annotation_file_name,
        specific_labels=specific_classes,
        include_label_borders=include_label_borders,
        label_mode=label_mode,
        keep_mode=keep_mode,
        floating_point_labels=floating_point_labels,
        transforms=transforms
    )

    if not test_dataset._labels:
        return torch.as_tensor([]), torch.as_tensor([])

    test_data_loader = initialize_data_loader(
        batch_size=4,
        dataset=test_dataset,
        mode='test',
        data_sampler='undersampling' if undersampling else None,
        data_loader_settings={'workers': 8}
    )

    if undersampling:
        print('Test dataset created with Undersamping Sampler! Not all data points are sampled using this setting.')

    model = model.eval()

    predictions = []
    targets = []
    for data in test_data_loader:
        batch, labels, *other_data = data
        # set the device of the batch and labels
        batch, labels = set_data_device(data=batch, labels=labels, device=device)

        # predict the target
        prediction = model(batch)

        # store predictions
        predictions.append(prediction)

        # store targets
        targets.append(labels.flatten())
    
    return torch.cat(predictions).detach().cpu(), torch.cat(targets).detach().cpu()


def save_confusion_matrix(save_path: str, predictions: torch.Tensor, targets: torch.Tensor,
    x_label: str = 'Predicted Values', y_label: str = 'Actual Values', file_name: str = 'confusion_matrix.png'):
    # take max of second dimension of predictions (classification)
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        predictions = torch.argmax(predictions, dim=1).reshape(predictions.shape[0], 1)
    
    if len(file_name.split('.')) > 1:
        file_name = file_name[:-len(file_name.split('.')[-1]) - 1]
    
    predictions = predictions.reshape(shape=targets.shape)

    predictions = torch.round(torch.clip(predictions, min=0.0, max=torch.unique(torch.cat((targets * 1.0, predictions * 1.0))).max()))

    sklearn_cf, _ = confusion_matrix(
        predictions=predictions, targets=targets,
        output_path=os.path.join(save_path, 'confusion_matrix.png' if not file_name else file_name + '.png'),
        id_class_name_map=None,
        exists_ok=True,
        # x_label=x_label, y_label=y_label
    )

    sklearn_confusion_matrix_to_txt(
        confusion_matrix=sklearn_cf, output_file_path=os.path.join(save_path, 'confusion_matrix.txt' if not file_name else file_name + '.txt')
    )



def calculate_and_save_accuracy(
    predictions: torch.Tensor, targets: torch.Tensor, save_path: str = None, header: List or str = 'Accuracy', file_name: str = 'accuracy.txt'):
    if isinstance(header, str):
        header = [header]
    
    # save test accuracy
    accuracy = calculate_accuracy(
        model_outputs=predictions, targets=targets, calc_mean=True
    )

    if save_path:
        if len(file_name.split('.')) > 1:
            file_name = file_name[:-len(file_name.split('.')[-1]) - 1]

        # save accuracy to txt
        lines = [['{:.4f}'.format(accuracy)]]
        str_list_to_txt(
            lines=lines, header=header,
            output_file_path=os.path.join(save_path, 'accuracy.txt' if not file_name else file_name + '.txt')
        )
    
    return accuracy


def calculate_and_save_prediction_target_relations(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    save_path: str = None,
    file_name: str = 'prediction_target_relations'
):
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        predictions = torch.argmax(predictions, dim=1).reshape(predictions.shape[0], 1)
    else:
        # check if regression prediction is greater than the max target -> if so, change pred to max target
        predictions = torch.where(
            predictions > targets.max(),
            torch.ones(size=predictions.size()) * targets.max(),
            predictions
        )
        predictions = torch.where(
            predictions < targets.min(),
            torch.ones(size=predictions.size()) * targets.min(),
            predictions
        )

        predictions = torch.round(predictions)

    # flatten predictions and targets
    predictions = predictions.flatten()
    targets = targets.flatten()

    # calculate per label accuracy
    if torch.is_floating_point(predictions):
        uniques = torch.unique(targets, sorted=True).tolist()
    else:
        uniques = torch.unique(torch.cat((targets, predictions)), sorted=True).tolist()

    unique_predictions_indices = {
        prediction: torch.where(predictions == prediction)[0].tolist() for prediction in uniques
    }

    precision = {
        p: calculate_accuracy(
            predictions[unique_predictions_indices[p]],
            targets[unique_predictions_indices[p]]
         ).item() for p in unique_predictions_indices if unique_predictions_indices[p]
    }

    prediction_mean_deviation = {
        p: torch.mean(
            (predictions[unique_predictions_indices[p]] - targets[unique_predictions_indices[p]]).to(dtype=torch.float32)
        ).item() for p in unique_predictions_indices if unique_predictions_indices[p]
    }

    prediction_std_deviation = {
        p: torch.std(
            (predictions[unique_predictions_indices[p]] - targets[unique_predictions_indices[p]]).to(dtype=torch.float32)
        ).item() if len(unique_predictions_indices[p]) > 1 else 0
        for p in unique_predictions_indices if unique_predictions_indices[p]
    }

    unique_target_indices = {
        target: torch.where(targets == target)[0].tolist() for target in uniques
    }

    recall = {
        t: calculate_accuracy(
            predictions[unique_target_indices[t]],
            targets[unique_target_indices[t]]
        ).item() for t in unique_target_indices
    }

    target_mean_deviation = {
        t: torch.mean(
            (predictions[unique_target_indices[t]] - targets[unique_target_indices[t]]).to(dtype=torch.float32)
        ).item() for t in unique_target_indices if unique_target_indices[t]
    }

    target_std_deviation = {
        t: torch.std(
            (predictions[unique_target_indices[t]] - targets[unique_target_indices[t]]).to(dtype=torch.float32)
        ).item() if len(unique_target_indices[t]) > 1 else 0
        for t in unique_target_indices if unique_target_indices[t]
    }

    target_rmse = {
        t: torch.sqrt(torch.sum(
            (predictions[unique_target_indices[t]] - targets[unique_target_indices[t]]).to(dtype=torch.float32)**2) / len(unique_target_indices[t])
        ).item() for t in unique_target_indices if unique_target_indices[t]
    }

    target_overall_rmse = torch.sqrt(torch.sum((predictions - targets)**2) / predictions.shape[0]).item()

    target_mae = {
        t: (torch.sum(torch.abs(
            (predictions[unique_target_indices[t]] - targets[unique_target_indices[t]]).to(dtype=torch.float32)
            )) / len(unique_target_indices[t])
        ).item() for t in unique_target_indices if unique_target_indices[t]
    }

    target_mse = {
        t: (torch.sum(
            (predictions[unique_target_indices[t]] - targets[unique_target_indices[t]]).to(dtype=torch.float32)**2
            ) / len(unique_target_indices[t])
        ).item() for t in unique_target_indices if unique_target_indices[t]
    }

    # f1 score
    # micro (every single prediction equally weighted)
    micro_f1_score = f1_score(y_true=targets.flatten().numpy(), y_pred=predictions.flatten().numpy(), average='micro')

    # macro (every single class equally weighted)
    macro_f1_score = f1_score(y_true=targets.flatten().numpy(), y_pred=predictions.flatten().numpy(), average='macro')

    #
    weighted_f1_score = f1_score(y_true=targets.flatten().numpy(), y_pred=predictions.flatten().numpy(), average='weighted')


    summary = dict(
        precision=precision,
        prediction_mean_deviation=prediction_mean_deviation,
        prediction_std_deviation=prediction_std_deviation,
        recall=recall,
        target_mean_deviation=target_mean_deviation,
        target_std_deviation=target_std_deviation,
        target_rmse=target_rmse,
        target_overall_rmse=target_overall_rmse,
        target_mae=target_mae,
        target_mse=target_mse,
        micro_f1_score=micro_f1_score,
        macro_f1_score=macro_f1_score,
        weighted_f1_score=weighted_f1_score
    )

    if save_path:
        if len(file_name.split('.')) > 1:
            file_name = file_name[:-len(file_name.split('')[-1])]
        # save calculations to txt
        dict_to_txt(
            summary, output_file_path=os.path.join(save_path, file_name + '.txt')
        )
    
    return summary


def calculate_per_label_regression_accuracy(targets, predictions, scale: str = 'linear1'):
    def _calculate_linear_accuracy(target, prediction):
        return max(-abs(target  - prediction), -1) + 1

    calc_method = None
    if scale == 'linear1':
        calc_method = _calculate_linear_accuracy
    else:
        raise NotImplementedError(f'Method: {calc_method} is not implemented yet.')

    targets = [t.item() for t in targets]
    predictions = [p.item() for p in predictions]

    # targets and predictions to sorted dict
    t_min, t_max = min(targets), max(targets)

    # clip predictions at the borders
    predictions = [max(min(p, t_max), t_min) for p in predictions]

    d_predictions = {t: [] for t in targets}

    for t, p in zip(targets, predictions):
        d_predictions[t].append(calc_method(t, p))
    
    # calculate the per label accuracies
    accuracies = {
        p: sum(d_predictions[p]) / len(d_predictions[p]) for p in d_predictions
    }

    weighted_overall_acc = sum([acc for t in d_predictions for acc in d_predictions[t]]) / sum([len(d_predictions[t]) for t in d_predictions])

    return accuracies, weighted_overall_acc


def default_evaluation(
    save_path: str,
    model: torch.nn.Module,
    model_type: str,  # regression or classification
    data_dir: str,
    test_annotation_file_name: str,
    model_run_path: str,
    model_name: str,
    dataset_year: str,
    camera_name: str,
    specific_classes: str,
    include_label_borders: bool = False,
    produce_reproducible_path: bool = True,
    save_outputs: bool = True,
    device: torch.device = torch.device('cpu'),
    # for dataset merger (if multiple test annotation files are to be used - separated by '+')
    label_mode: str = 'mean',
    keep_mode: str = 'intersection',
    transforms = None
) -> Tuple[float, float, str, torch.Tensor, torch.Tensor]:

    full_save_path = save_path

    if isinstance(specific_classes, str):
        specific_classes = '_'.join([str(label) for label in specific_classes.split('_') if label.isdigit()])

    if produce_reproducible_path:
        timestamp = os.path.basename(model_run_path)
        if include_label_borders:
            _specific_classes = '_'.join([str(specific_classes), 'with_label_borders'])
        full_save_path = os.path.join(save_path, str(model_type), str(model_name), str(dataset_year), camera_name, str(_specific_classes), timestamp)

    if save_outputs and not os.path.isdir(full_save_path):
        os.makedirs(full_save_path)
    
    ret_struc = lambda: dict(
        balanced=None,
        unbalanced=None
    )

    predictions = ret_struc()
    targets = ret_struc()
    accuracies = ret_struc()

    for t in [
        #('balanced', True),
        ('unbalanced', False)
        ]:
        key, is_balanced = t

        suffix = key + '_'

        output_file_path = os.path.join(full_save_path, suffix + 'output.txt')
        confusion_matrix_file_name = suffix + 'confusion_matrix'
        accuracy_file_name = suffix + 'accuracy'
        prediction_target_relations_file_name = suffix + 'prediction_target_relations'
        cohen_kappa_file_name = suffix + 'cohen_kappa.txt'
        soft_accuracy_file_name = suffix + 'soft_accuracy.txt'

        predictions[key], targets[key] = eval_model_on_test_dataset(
            model=model,
            data_dir=data_dir,
            data_year=dataset_year,
            camera_name=camera_name,
            test_annotation_file_name=test_annotation_file_name,
            specific_classes=specific_classes,
            include_label_borders=include_label_borders,
            device=device,
            label_mode=label_mode,
            keep_mode=keep_mode,
            floating_point_labels=True if model_type == 'regression' else False,
            undersampling=True if is_balanced else False,
            transforms=transforms if transforms else None
        )

        if not torch.any(predictions[key]) or not torch.any(targets[key]):
            continue

        # rounded targets for confusion matrix and accuracy calculations
        rounded_targets = targets[key]

        if torch.is_floating_point(targets[key]):
            rounded_targets = torch.round(targets[key]).to(dtype=torch.uint8)

        if save_outputs:
            # save best model and configuration into eval folder (copy of run folder)

            shutil.copy(os.path.join(model_run_path, 'best_model.pth'), os.path.join(full_save_path, 'best_model.pth'))
            shutil.copy(os.path.join(model_run_path, 'model_configuration.pth_info'), os.path.join(full_save_path, 'model_configuration.pth_info'))

            # save predictions and labels into txt file
            tensors_to_txt(
                tensors=[predictions[key], targets[key]],
                output_file_path=output_file_path,
                tensor_names=['Predictions', 'Labels']
            )

            ############################
            # Produce confusion matrix #
            ############################
            save_confusion_matrix(
                save_path=full_save_path, predictions=predictions[key], targets=rounded_targets, file_name=confusion_matrix_file_name
            )

        if save_outputs:
            from sklearn.metrics import cohen_kappa_score

            weights = 'linear'

            if len(predictions[key].shape) == 2 and predictions[key].shape[1] > 1:
                # classification
                y1 = torch.argmax(predictions[key], dim=1)
            else:
                y1 = torch.clip(torch.round(predictions[key]), min=0.00, max=rounded_targets.max())

            k = cohen_kappa_score(
                y1=y1, y2=rounded_targets, weights=weights
            )

            str_list_to_txt(
                lines=[[str(k)]],
                header=["Cohen's Kappa"],
                output_file_path=os.path.join(full_save_path, cohen_kappa_file_name)
            )
        
        if save_outputs and not (len(predictions[key].shape) == 2 and predictions[key].shape[1] > 1):
            soft_accuracy, overall_soft_acc = calculate_per_label_regression_accuracy(
                targets=rounded_targets, predictions=predictions[key], scale='linear1'
            )

            soft_accuracy.update({'overall_weighted': overall_soft_acc})
            dict_to_txt(
                soft_accuracy,
                output_file_path=os.path.join(full_save_path, soft_accuracy_file_name)
            )

        accuracies[key] = calculate_and_save_accuracy(
            predictions=predictions[key], targets=rounded_targets,
            save_path=full_save_path if save_outputs else None,
            header='Accuracy', file_name=accuracy_file_name
        )

        calculate_and_save_prediction_target_relations(
            predictions=predictions[key], targets=rounded_targets,
            save_path=full_save_path if save_outputs else None,
            file_name=prediction_target_relations_file_name
        )
        
    
    return accuracies, predictions, targets, full_save_path


if __name__ == '__main__':
    set_seed(42)

    specific_expert = 'rater1'
    # path of your model runs
    run_path = r'./runs'
    # evaluation save path
    eval_save_dir = r'./evaluations'

    data_directory = r'path/to/data'

    run_path = os.path.join(run_path, specific_expert)
    eval_save_dir = os.path.join(eval_save_dir, specific_expert)

    # options for merger if multiple test files are available
    label_mode = 'mean'
    keep_mode = 'intersection'

    device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')

    # ranking separated into
    # regression / classification
    #   model_name
    #      dataset year
    #          specific classes
    ranking = dict()

    model_paths = load_run_paths(root_path=run_path)
    for model, config, m_type, run_path, model_name, year, camera_name, specific_classes in model_evaluation_iterator(run_paths=model_paths, device=device):
        config_model_parameters = config['model_parameters']
        data_dir = config['function_parameters']['data_dir']
        data_dir = data_directory if data_directory else data_dir
        test_annotation_file_name = config['function_parameters']['test_annotation_file_name']
    
        include_label_borders = config['function_parameters']['include_label_borders']

        if isinstance(test_annotation_file_name, list) or isinstance(test_annotation_file_name, tuple):
            if len(test_annotation_file_name) > 1:
                print(f'Multiple test files found. Please check the annotation file input, got {test_annotation_file_name}.')
            # pick first
            test_annotation_file_name = test_annotation_file_name[0]

        accuracies, _, _, eval_path = default_evaluation(
            save_path=eval_save_dir,
            model=model,
            model_type=m_type,
            data_dir=data_dir,
            test_annotation_file_name=test_annotation_file_name,
            model_run_path=run_path,
            model_name=model_name,
            dataset_year=year,
            camera_name=camera_name,
            specific_classes=specific_classes,
            include_label_borders=include_label_borders,
            device=device,
            # for dataset merger (if multiple test annotation files are to be used - separated by '+')
            label_mode=label_mode,
            keep_mode=keep_mode,
            transforms=None
        )
        # add to ranking

        if m_type not in ranking:
            ranking[m_type] = dict()
        
        if model_name not in ranking[m_type]:
            ranking[m_type][model_name] = dict()
        
        if year not in ranking[m_type][model_name]:
            ranking[m_type][model_name][year] = dict()

        if specific_classes not in ranking[m_type][model_name][year]:
            ranking[m_type][model_name][year][specific_classes] = dict(
                balanced_accuracies=[],
                unbalanced_accuracies=[]
            )
        
        if accuracies['balanced'] is not None:        
            # add result
            ranking[m_type][model_name][year][specific_classes]['balanced_accuracies'].append(dict(
                eval_path=eval_path,
                value=accuracies['balanced'].item()
            ))

        if accuracies['unbalanced'] is not None:
            ranking[m_type][model_name][year][specific_classes]['unbalanced_accuracies'].append(dict(
                eval_path=eval_path,
                value=accuracies['unbalanced'].item()
            ))

        # sort accuracy
        balanced_accuracy_sorted_ids = [i[0] for i in sorted(enumerate(ranking[m_type][model_name][year][specific_classes]['balanced_accuracies']), key=lambda x:x[1]['value'], reverse=True)]
        unbalanced_accuracy_sorted_ids = [i[0] for i in sorted(enumerate(ranking[m_type][model_name][year][specific_classes]['unbalanced_accuracies']), key=lambda x:x[1]['value'], reverse=True)]

        # use indices to resort
        ranking[m_type][model_name][year][specific_classes]['balanced_accuracies'] = [
            ranking[m_type][model_name][year][specific_classes]['balanced_accuracies'][_id] for _id in balanced_accuracy_sorted_ids
        ]

        ranking[m_type][model_name][year][specific_classes]['unbalanced_accuracies'] = [
            ranking[m_type][model_name][year][specific_classes]['unbalanced_accuracies'][_id] for _id in unbalanced_accuracy_sorted_ids
        ]


        # Some other custom evaluations

    # create ranking file for accuracies
    dict_to_txt(
        dictionary=ranking, output_file_path=os.path.join(eval_save_dir, 'ranking.txt')
    )
