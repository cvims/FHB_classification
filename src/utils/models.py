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
Configurations for different predefined models
"""
# =============================================================================
# Imports
# =============================================================================
from typing import Any, Dict
import torch.nn as nn
import torchvision.models as models


def load_efficientnet(name: str, output_classes: int,
                      pretrained: bool = True, freeze_layers: bool = True,
                      **kwargs) -> Any:
    """
    Loads the efficinet net model from torchvision
    :param name: Name (size) of efficientnet model b0, b1, ...
    :param pretrained: boolean flag. Yes for pretrained, no for untrained.
    :param freeze_layers: boolean flag. If true then all layers except the output layer will be frozen.
    :param output_classes: number of output classes
    """
    model = None
    if name == 'b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None, **kwargs)
    elif name == 'b1':
        model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT if pretrained else None, **kwargs)
    elif name == 'b2':
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT if pretrained else None, **kwargs)
    elif name == 'b3':
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT if pretrained else None, **kwargs)
    elif name == 'b4':
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT if pretrained else None, **kwargs)
    elif name == 'b5':
        model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.DEFAULT if pretrained else None, **kwargs)
    elif name == 'b6':
        model = models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.DEFAULT if pretrained else None, **kwargs)
    elif name == 'b7':
        model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT if pretrained else None, **kwargs)
    
    if not model:
        raise Exception(f'No model with size {name} available.')
    
    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False

    # change the output layer
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, output_classes)

    return model


def load_resnet(
    name: str or int, output_classes: int,
    pretrained: bool = True, freeze_layers: bool = True,
    **kwargs) -> Any:
    """
    Loads the resnet model from torchvision
    :param name: Name (size) of resnet model 18, 34, 50, 101, 152
    :param pretrained: boolean flag. Yes for pretrained, no for untrained.
    :param freeze_layers: boolean flag. If true then all layers except the output layer will be frozen.
    :param output_classes: number of output classes
    """
    resnet_size = int(name)
    model = None
    if resnet_size == 18:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None, **kwargs)
    elif resnet_size == 34:
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None, **kwargs)
    elif resnet_size == 50:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None, **kwargs)
    elif resnet_size == 101:
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT if pretrained else None, **kwargs)
    elif resnet_size == 152:
        model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT if pretrained else None, **kwargs)

    if not model:
        raise Exception(f'No model with size {name} available.')
    
    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False

    # change the output layer
    model.fc = nn.Linear(model.fc.in_features, output_classes)

    return model


def load_model(name: str, output_classes: int, config: Dict) -> Any:
    if name == 'efficientnet':
        return load_efficientnet(output_classes=output_classes, **config)
    elif name == 'resnet':
        return load_resnet(output_classes=output_classes, **config)
    else:
        raise NotImplementedError(f'No configuration for model with name "{name}" found.')
