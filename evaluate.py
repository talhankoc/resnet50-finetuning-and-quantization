import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.quantization import resnet50
from utils.quantization import _replace_relu, quantize_model

from indoor_dataset import IndoorDataset
from utils.helpers import get_file_names

NUM_LABELS = 67
IMAGES_DIR = "data/Images/"
VAL_IMAGES_FN = "data/Test Images.txt"
MODEL_WEIGHTS_PATH = "model_params/finetuned_model.pt"


def evaluate(model, dataloader, criterion):
    """This function prints model accuracy and loss on the given dataset"""
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        t0 = time.time()
        for i, (inputs, labels) in enumerate(dataloader):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    t1 = time.time()
    time_elapsed = t1 - t0
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))


def evaluate_speed(model, dataloader):
    """This function evaluates only the speed of the given model"""
    with torch.no_grad():
        t0 = time.time()
        for inputs, _ in dataloader:
            model(inputs)
    t1 = time.time()
    time_elapsed = t1 - t0
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Inference speed was {:.5f}s per sample at batch size {:d}'.format(
        time_elapsed / float(len(dataloader.dataset)), dataloader.batch_size))


def load_finetuned_model():
    """Loads fine-tuned model. Assumes you have run the `train_resnet` module to generate saved model params."""
    model = resnet50(pretrained=False, quantize=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_LABELS)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
    return model


def quantize_model_helper(model, dummy_input_data):
    """This function quantizes the given model """
    ### COPY --- this has been taken from torchvision.models.quantization.resnet and modified
    _replace_relu(model)
    backend = 'fbgemm'
    quantize_model(model, backend, dummy_input_data)
    ### END COPY
    print('Model quantized succesfully!')
    return model


def init_val_dataloader(batch_size):
    # Get file names for train and test datasets
    val_file_names = get_file_names(VAL_IMAGES_FN)
    val_dataset = IndoorDataset(file_names=val_file_names)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    print(f'Validation dataloader initialized. Size {len(val_dataset)}')
    return val_data_loader


if __name__ == '__main__':
    val_data_loader = init_val_dataloader(batch_size=32)
    criterion = nn.CrossEntropyLoss()

    # evaluate non-quantized model
    model = load_finetuned_model()
    evaluate(model, val_data_loader, criterion)
    evaluate_speed(model, val_data_loader)

    # evaluate after quantizing model
    model_quant = quantize_model_helper(model, dummy_input_data=val_data_loader.dataset[0][0].unsqueeze(0))
    evaluate(model, val_data_loader, criterion)
    evaluate_speed(model, val_data_loader)

