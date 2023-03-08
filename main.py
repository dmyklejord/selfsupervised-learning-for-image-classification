
# Note: The models are not true to their respective papers, but are close.
# Choices were made so that they could be run on single-GPU consumer hardware.

import torch
from torch import nn
import torchvision
import copy
import numpy as np
import os

from lightly.models.modules import SimCLRProjectionHead, MoCoProjectionHead
from lightly.models.modules import BYOLProjectionHead, BYOLPredictionHead
from lightly.models.utils import deactivate_requires_grad
from sklearn.manifold import TSNE

import helper_train
import helper_evaluate
import helper_data

# Hyperparameters
RANDOM_SEED = 123
LEARNING_RATE = 0.1
BATCH_SIZE = 128
NUM_EPOCHS = 10

# Set to run on mps if available (i.e. Apple's GPU).
# mps is a new pytorch feature, so we check that
# it's also available with the user's pytorch install.
DEVICE = 'mps' if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else 'cpu'

helper_evaluate.set_deterministic
helper_evaluate.set_all_seeds(RANDOM_SEED)

# Getting the data:
# # For CIFAR10:
# train_loader, test_loader = helper_data.get_dataloaders_cifar10(batch_size=BATCH_SIZE)
# class_names = ('plane', 'car', 'bird', 'cat',
#             'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# For other data:
# The directory should contain folders of images, with each folder
# having images of a certain class. Example: 2 folders for 2 classes.
# The folder names should be the class names.
data_location=('data')
train_loader, test_loader = helper_data.get_dataloaders(data_location, batch_size=BATCH_SIZE)
class_names = [f.name for f in os.scandir(data_location) if f.is_dir()]

# The models themselves:
class SimCLR(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(512, 512, 128)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

class MoCo(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = MoCoProjectionHead(512, 512, 128)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        query = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(query)
        return query

    def forward_momentum(self, x):
        key = self.backbone_momentum(x).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key

class BYOL(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(512, 1024, 256)
        self.prediction_head = BYOLPredictionHead(256, 1024, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    # i.e. the online branch
    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    # i.e. the target branch
    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z

# Building the backbone. Here is where you can change it to whatever you
# want, for example a resent50. weights=DEFAULT initializes to ImageNet1k weights:
resnet = torchvision.models.resnet18(weights='DEFAULT')
backbone = nn.Sequential(*list(resnet.children())[:-1]) # removes FC layer

# In this case, we want to use MoCo with a SGD optimizer:
model = MoCo(backbone).to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Train the model, returns a dict of the training loss. Change the
# function call to train the model you want.
model_name=f'{model.__class__.__name__}_{NUM_EPOCHS}EP_{BATCH_SIZE}BS_{LEARNING_RATE}LR'
log_dict = helper_train.train_moco(num_epochs=NUM_EPOCHS, model=model,
                    optimizer=optimizer, device=DEVICE,
                    train_loader=train_loader,
                    save_model=model_name,
                    logging_interval=2,
                    save_epoch_states=False)

# Saving the log of training loss to a csv
import csv
w = csv.writer(open(model_name+'_LossLog.csv', "w"))
for key, val in log_dict.items():
    w.writerow([key, val])

# Inference:
# Passing images through the trained backbone to get their embeddings/features/latent_space etc.
train_X, train_y, test_X, test_y, test_images = helper_evaluate.get_features(model, train_loader, test_loader, DEVICE)
np.savez(model_name+'_features', train_X=train_X, train_y=train_y, test_X=test_X, test_y=test_y, test_images=test_images)

# Trains a linear classifier on the data to determine accuracy and makes a confusion matrix.
# Lets us see exactly what's going on under the hood, vs using sklearn.
# returns dict of training and final losses and accuracies.
log_dict = helper_evaluate.lin_eval(train_X, train_y, test_X, test_y, model_name, classes, DEVICE='cpu')

# BUT, it's slow, so we can use sklearn instead if we want:
# pred_labels = helper_evaluate.linear_classifier(train_X, train_y, test_X, test_y)
# confusion_matrix, accuracy = helper_evaluate.make_confusion_matrix(pred_labels, test_y, len(class_names))
# helper_evaluate.visualize_confusion_matrix(confusion_matrix, accuracy, class_names, model_name)

# TSNE analysis and visualization:
tsne_xtest = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=20, n_iter=1000).fit_transform(test_X)
helper_evaluate.visualize_tsne(model_name, tsne_xtest, class_names, test_y, close_fig=True)

# Visualize TSNE with predicted labels:
pred_labels = helper_evaluate.linear_classifier(train_X, train_y, test_X, test_y)
helper_evaluate.visualize_hover_images(model_name, tsne_xtest, test_images, pred_labels, class_names, test_y, showplot=True)

'''
# Extra functions that may be handy:

pred_labels = helper_evaluate.kmeans_classifier_2class(test_X, test_y)
pred_labels = helper_evaluate.kmeans_classifier(test_X, k=100)
pred_labels = helper_evaluate.knn_classifier(train_X, train_y, test_X, test_y, k=100)

# For saving time with pre-trained model:
model.load_state_dict(torch.load(f'{model_name}.pt')) # to load a pre-trained model
features = np.load(model_name+'_features.npz')          # to load the embedded space that you've already found
features = np.load('Moco_Lv8_SL_10EP_128BS_0.1LR_4096MB_features.npz')
train_X, train_y, test_X, test_y, test_images = features['train_X'], features['train_y'], features['test_X'], features['test_y'], features['test_images']   # Gets the embedded space into a usable format.
'''
