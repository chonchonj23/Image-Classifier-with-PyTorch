
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
import argparse


parser = argparse.ArgumentParser(description='Neural Network for Image Classifier')
parser.add_argument('--data_dir', metavar='', type=str, default="./flowers", help='please input the directory to save model results')
parser.add_argument('--save_dir', metavar='', type=str, default="./checkpoint_py.pth", help='please input the directory to save model results')
parser.add_argument('--arch', metavar='', type=str, default='vgg19',  help='please choose a CNN network, such as vgg19 or densent121')
parser.add_argument('--gpu', metavar='', type=str, default='gpu', help='can choose GPU')
parser.add_argument('--learning_rate', metavar='', default=0.001, type=float, help='please input appropriate learing rate for model')
parser.add_argument('--dropout', metavar='', type=float, default=0.2, help='please input appropriate dropout rate for model')
parser.add_argument('--hidden_units_layer1', metavar='', type=int, default=4096, help='please input the number of hidden units in 1st hidden layer')
parser.add_argument('--hidden_units_layer2', metavar='', type=int, default=2048, help='please input the number of hidden units in 2nd hidden layer')
parser.add_argument('--epochs', metavar='', type=int, default=1, help='please input the number of epochs for model training' )
args = parser.parse_args()

#check user inputs
if ( args.data_dir == 'help' ):
    print('please input the directory to save model results, such as ./flowers')
    quit()

if ( args.save_dir == 'help' ):
    print('please input the directory to save model results, such as ./checkpoint_py.pth')
    quit()

if ( args.arch == 'help' ):
    print('please see below the 3 architectures available in this program')
    print('1. vgg19 (default)')
    print('2. densenet169')
    print('3. resnet101')
    quit()


arch_types = ["vgg19", "resnet101", "densenet169", "inception_v3"]
if args.arch not in arch_types:
    print("Error: Invalid architecture name")
    print('Please type \"python train.py --arch help\" for more information')
    quit()


if (not( args.learning_rate > 0 and args.learning_rate < 1 )):
    print('Learning Rate must be between 0 and 1')
    quit()

if args.hidden_units_layer1 <= 0:
    print('Hidden units must be greater than zero')
    quit()

if args.hidden_units_layer2 <= 0:
    print('Hidden units must be greater than zero')
    quit()

if args.epochs <= 0:
    print('Hidden units must be greater than zero')
    quit()



data_location = args.data_dir
path_saving = args.save_dir
arch_PreModel = args.arch
# processor = args.gpu
learn_rate = args.learning_rate
dropout_rate = args.dropout
hidden_layer1 = args.hidden_units_layer1
hidden_layer2 = args.hidden_units_layer2
epochs = args.epochs

data_dir = data_location
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


test_valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_valid_transforms)
Valid_data = datasets.ImageFolder(valid_dir, transform=test_valid_transforms)


trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
validloader = torch.utils.data.DataLoader(Valid_data, batch_size=64)


if args.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model_selection = {
        "vgg19": models.vgg19(pretrained=True),
        "densenet169": models.densenet169(pretrained=True),
        "resnet101": models.resnet101(pretrained=True),
        "inception_v3": models.inception_v3(pretrained=True)
}

model = model_selection.get(arch_PreModel)

classifier1 = None
optimizer = None

for param in model.parameters():
    param.requires_grad = False


if arch_PreModel == "vgg19":
    classifier1 = nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(25088, hidden_layer1)),
                    ('relu1', nn.ReLU()),
                    ('drop1', nn.Dropout(dropout_rate)),
                    ('fc2', nn.Linear(hidden_layer1, hidden_layer2)),
                    ('relu2', nn.ReLU()),
                    ('drop2', nn.Dropout(dropout_rate)),
                    ('fc3', nn.Linear(hidden_layer2, 102)),
                    ('output', nn.LogSoftmax(dim=1))
                     ]))
    model.classifier = classifier1
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)


elif arch_PreModel == "densenet169":
    classifier1 = nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(1664, hidden_layer1)),
                    ('relu1', nn.ReLU()),
                    ('drop1', nn.Dropout(dropout_rate)),
                    ('fc2', nn.Linear(hidden_layer1, hidden_layer2)),
                    ('relu2', nn.ReLU()),
                    ('drop2', nn.Dropout(dropout_rate)),
                    ('fc3', nn.Linear(hidden_layer2, 102)),
                    ('output', nn.LogSoftmax(dim=1))
                     ]))
    model.classifier = classifier1
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)


elif arch_PreModel == "resnet101":
    classifier1 = nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(2048, hidden_layer1)),
                    ('relu1', nn.ReLU()),
                    ('drop1', nn.Dropout(dropout_rate)),
                    ('fc2', nn.Linear(hidden_layer1, hidden_layer2)),
                    ('relu2', nn.ReLU()),
                    ('drop2', nn.Dropout(dropout_rate)),
                    ('fc3', nn.Linear(hidden_layer2, 102)),
                    ('output', nn.LogSoftmax(dim=1))
                     ]))
    model.fc = classifier1
    optimizer = optim.Adam(model.fc.parameters(), lr=learn_rate)


# elif arch_PreModel == "inception_v3":
#     classifier1 = nn.Sequential(OrderedDict([
#                     ('fc1', nn.Linear(2048, hidden_layer1)),
#                     ('relu1', nn.ReLU()),
#                     ('drop1', nn.Dropout(dropout_rate)),
#                     ('fc2', nn.Linear(hidden_layer1, hidden_layer2)),
#                     ('relu2', nn.ReLU()),
#                     ('drop2', nn.Dropout(dropout_rate)),
#                     ('fc3', nn.Linear(hidden_layer2, 102)),
#                     ('output', nn.LogSoftmax(dim=1))
#                      ]))
#     model.fc = classifier1
#     optimizer = optim.Adam(model.fc.parameters(), lr=learn_rate)



criterion = nn.NLLLoss()
model.to(device)


# below code is copied from workspace_utils.py, which is provided in the Introduction section

import signal
from contextlib import contextmanager
import requests

DELAY = INTERVAL = 4 * 60  # interval time in seconds
MIN_DELAY = MIN_INTERVAL = 2 * 60
KEEPALIVE_URL = "https://nebula.udacity.com/api/v1/remote/keep-alive"
TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token"
TOKEN_HEADERS = {"Metadata-Flavor":"Google"}

def _request_handler(headers):
    def _handler(signum, frame):
        requests.request("POST", KEEPALIVE_URL, headers=headers)
    return _handler

@contextmanager
def active_session(delay=DELAY, interval=INTERVAL):
    """
    Example:

    from workspace_utils import active session

    with active_session():
        # do long-running work here
    """
    token = requests.request("GET", TOKEN_URL, headers=TOKEN_HEADERS).text
    headers = {'Authorization': "STAR " + token}
    delay = max(delay, MIN_DELAY)
    interval = max(interval, MIN_INTERVAL)
    original_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _request_handler(headers))
        signal.setitimer(signal.ITIMER_REAL, delay, interval)
        yield
    finally:
        signal.signal(signal.SIGALRM, original_handler)
        signal.setitimer(signal.ITIMER_REAL, 0)


def keep_awake(iterable, delay=DELAY, interval=INTERVAL):
    """
    Example:

    from workspace_utils import keep_awake

    for i in keep_awake(range(5)):
        # do iteration with lots of work here
    """
    with active_session(delay, interval): yield from iterable


steps = 0
running_loss = 0
print_every = 10


with active_session():
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)

                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()


print("Our model: \n\n", model, '\n')
print("The state dict keys: \n\n", model.state_dict().keys())


# torch.save(model.state_dict(), 'checkpoint.pth')
# state_dict = torch.load('checkpoint.pth')
# print(state_dict.keys())

#  Save the checkpoint

model.class_to_idx = train_data.class_to_idx


torch.save({'Pre_train': arch_PreModel,
            'lr':learn_rate,
            'dropout':dropout_rate,
            'hidden_units_layer1': hidden_layer1,
            'hidden_units_layer2': hidden_layer2,
            'no_of_epochs':epochs,
            'state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx }, path_saving)
