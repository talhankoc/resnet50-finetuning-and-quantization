import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.quantization import resnet50

from indoor_dataset import IndoorDataset
from utils.helpers import get_file_names, train_model

NUM_LABELS = 67
IMAGES_DIR = "data/Images/"
TRAIN_IMAGES_FN = "data/Train Images.txt"
VAL_IMAGES_FN = "data/Test Images.txt"


def main():
    # Get file names for train and test datasets
    train_file_names = get_file_names(TRAIN_IMAGES_FN)
    val_file_names = get_file_names(VAL_IMAGES_FN)

    train_dataset = IndoorDataset(file_names=train_file_names)
    val_dataset = IndoorDataset(file_names=val_file_names)

    train_data_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    model = resnet50(pretrained=True, quantize=False)

    # Freeze all weights -- we will only fine-tune the last layer
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Since we init a new layer, it will be the only layer that has a gradient
    model.fc = nn.Linear(model.fc.in_features, NUM_LABELS)
    # The value here should be just parameters associated with the last fully connected layer
    params_to_optimize = [param for _, param in model.named_parameters() if param.requires_grad]
    optimizer = optim.SGD(params_to_optimize, lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    ########
    # COPY #  The code below was copied from PyTorch documentation and modified to fit this use case
    ########
    dataloaders = {'train': train_data_loader, 'val': val_data_loader}
    model_ft, hist = train_model(
        model,
        dataloaders,
        optimizer,
        criterion
    )
    # END COPY #

    torch.save(model_ft.state_dict(), 'model_params/finetuned_model.pt')


if __name__ == "__main__":
    main()




