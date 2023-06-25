import torch
from torchvision.transforms import transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR


# image channel means and stds
means, stds = torch.tensor([0.4297, 0.4841, 0.3906]), torch.tensor([0.2380, 0.2355, 0.2155])

efficientnet_configuration = dict(
    model_name='efficientnet',
    model_parameters = dict(
        name = 'b0',
        pretrained=True,
        freeze_layers=True
    ),
    unfreeze_epoch=5,
    epochs=100,
    batch_size=2,
    data_loader_settings=dict(
        workers=8,
        persistent_workers=False,
        pin_memory=False,
        shuffle=True,
    ),
    data_train_sampler='weighted_random',  # weighted_random or undersampling
    data_train_transforms=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=(1500, 500)),
            transforms.RandomRotation(degrees=5.0),
            transforms.RandomCrop(size=(1425, 475)),
            transforms.Normalize(mean=means, std=stds),
            transforms.RandomHorizontalFlip(p=0.5)
        ]
    ),
    data_val_sampler='undersampling',  # weighted_random or undersampling
    data_val_transforms=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=(1500, 500))
        ]),
    optimizer=Adam,
    optimizer_kwargs=dict(),
    learning_rate=0.0001,
    lr_scheduler=MultiStepLR,
    lr_scheduler_kwargs=dict(
        milestones = [15]
    ),
    early_stopping=dict(
        patience=10,
        delta=0.01
    )
)