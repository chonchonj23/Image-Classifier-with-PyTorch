
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
import argparse
from PIL import Image


parser = argparse.ArgumentParser(description='load checkpoint of a the model trained in train.py')
parser.add_argument('--arch', metavar='', type=str, default='vgg19', help='please choose a CNN network, such as vgg19, resnet101, or densent121')
parser.add_argument('--checkpoint', metavar='', type=str, default="checkpoint_py.pth", help='Checkpoint of trained model for predictions.')
parser.add_argument('--image', metavar='', type=str, default="flowers/valid/28/image_05272.jpg", help='Location of image, e.g. flowers/test/class/image')
parser.add_argument('--topk', metavar='', type=int, default=5, help='Select number of classes you wish to see in descending order.')
parser.add_argument('--json', metavar='', action='store', type=str, default='cat_to_name.json', help='Provide name of json file for mapping of categories to real names')
parser.add_argument('--gpu', metavar='', type=str, default='gpu', help='can choose GPU')
parser.add_argument('--hidden_units_layer1', metavar='', type=int, default=4096, help='please input the number of hidden units in 1st hidden layer')
parser.add_argument('--hidden_units_layer2', metavar='', type=int, default=2048, help='please input the number of hidden units in 2nd hidden layer')
parser.add_argument('--dropout', metavar='', type=float, default=0.2, help='please input appropriate dropout rate for model')
args = parser.parse_args()


#check user inputs
if ( args.arch == 'help' ):
    print('please see which of the below 3 architectures has been used in the model you would like to load')
    print('1. vgg19 (default)')
    print('2. densenet169')
    print('3. resnet101')
    quit()


arch_types = ["vgg19", "resnet101", "densenet169"]
if args.arch not in arch_types:
    print("Error: Invalid architecture name")
    print('Please type \"python train.py --arch help\" for more information')
    quit()


if ( args.checkpoint == 'help' ):
    print('please check workspace directory to see the checkpoint file name')
    print('file name could be one of the following (depending on how you save the file in previous training')
    print('checkpoint_vgg19.pth')
    print('checkpoint_resnet101.pth')
    print('checkpoint_densenet169.pth')
    quit()


if args.hidden_units_layer1 <= 0:
    print('Hidden units must be greater than zero')
    print('This should match the parameter used in the trained model you are trying to load')
    quit()

if args.hidden_units_layer2 <= 0:
    print('Hidden units must be greater than zero')
    print('This should match the parameter used in the trained model you are trying to load')
    quit()


arch_PreModel = args.arch
checkpoint_path = args.checkpoint
image_path = args.image
topk = args.topk
arch_PreModel = args.arch
hidden_layer1 = args.hidden_units_layer1
hidden_layer2 = args.hidden_units_layer2
dropout_rate = args.dropout


json_path = args.json
with open(json_path, 'r') as file:
    cat_to_name = json.load(file)



if args.gpu== "gpu":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"



def load_checkpoint(filepath):
#     checkpoint = torch.load(filepath)
    model_load.load_state_dict(checkpoint['state_dict'])
    print("Pre-trained model: {}".format(checkpoint['Pre_train']))
    return model_load


# if arch_PreModel == 'vgg16':
#     model_load = models.vgg16(pretrained=True)
# else:
#     model_load = models.vgg19(pretrained=True)


model_selection = {
        "vgg19": models.vgg19(pretrained=True),
        "densenet169": models.densenet169(pretrained=True),
        "resnet101": models.resnet101(pretrained=True)
}
model_load = model_selection.get(arch_PreModel)

for param in model_load.parameters():
    param.requires_grad = False



checkpoint = torch.load(checkpoint_path)

if checkpoint['Pre_train'] == 'vgg19':
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
    model_load.classifier = classifier1

elif checkpoint['Pre_train'] == 'densenet169':
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
    model_load.classifier = classifier1

else:
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
    model_load.fc = classifier1



model_load = load_checkpoint(checkpoint_path)
print(model_load)


model_load.to(device)


mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model

    im = Image.open(image)

    if im.size[0] > im.size[1]:
        im.thumbnail((10000000, 256)) # restrict height to 256 when width > height
    else:
        im.thumbnail((256, 10000000)) # restrict width to 256 when width < height

    left_margin = (im.size[0]-224)/2
    bottom_margin = (im.size[1]-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224

    im = im.crop((left_margin, bottom_margin, right_margin, top_margin))

    np_image = np.array(im)/255

    np_image_nl = ( np_image - mean ) / std

    np_image_t = np_image_nl.transpose((2, 0, 1))

    return np_image_t



def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    if title:
        plt.title(title);

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax



def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    Image_np = process_image(image_path)

#     image_pt = torch.from_numpy(Image_np).type(torch.FloatTensor)
    image_pt = torch.from_numpy(Image_np).float().to(device)

    logps = model.forward(torch.tensor(image_pt.unsqueeze_(0)))

    ps = torch.exp(logps.cpu())

    probs, classes = ps.topk(5, dim=1)


    checkpoint_load = torch.load(checkpoint_path)

    idx_to_class_reverse = {value: key for key, value in checkpoint_load['class_to_idx'].items()}

    Cat_keys = [idx_to_class_reverse.get(key) for key in classes.numpy().tolist()[0]]

    Category_name = [cat_to_name.get(key) for key in Cat_keys]

    df_barchart = pd.DataFrame({'': Category_name, 'Probability': probs.detach().numpy().tolist()[0]})
    df_barchart = df_barchart.sort_values(by='Probability', ascending=False)

#     plt.figure(figsize = (3,5))
#     ax1 = plt.subplot(2,1,1)

#     chart_title = df_barchart[''][0]
#     imshow(Image_np, ax=ax1, title = chart_title);

#     print("\n")
#     sns.barplot(x=probs.detach().numpy().tolist()[0], y=Category_name, color=sns.color_palette()[0], ax=plt.subplot(2,1,2));
#     plt.show()


    print(df_barchart)
    print("\n")
    print("The predicted class is therefore: {}".format(df_barchart[''][0]))
    print("Its associated probability is: {}".format(df_barchart['Probability'][0]))

predict(image_path, model_load)
