import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import random


import pandas as pd
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Spectral10, Spectral5, Turbo256

from io import BytesIO
from PIL import Image
import base64


def set_deterministic():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_deterministic(True)

def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader

def inference(loader, model, device):
    feature_vector = []
    labels_vector = []
    image_vector = []
    for step, (x_query, _, y) in enumerate(loader):

        x_query = x_query.to(device)

        # get encoding
        with torch.no_grad():
            query = model.backbone(x_query).flatten(start_dim=1)

        query = query.detach()

        feature_vector.extend(query.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        image_vector.extend(x_query.cpu().detach().numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    image_vector = np.array(image_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector, image_vector

def get_features(model, train_loader, test_loader, device):
    train_X, train_y, _ = inference(train_loader, model, device)
    test_X, test_y, test_images = inference(test_loader, model, device)
    return train_X, train_y, test_X, test_y, test_images


class LogisticRegression(nn.Module):
    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()

        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.model(x)


def train_logistic_regression(loader, model, criterion, optimizer, DEVICE):
    loss_epoch = 0
    accuracy_epoch = 0

    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        # if step % 100 == 0:
        #     print(
        #         f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t Accuracy: {acc}"
        #     )

    return loss_epoch, accuracy_epoch


def test_logistic_regression(loader, model, criterion, optimizer, DEVICE):
    loss_epoch = 0
    accuracy_epoch = 0
    predicted_labels = []
    true_labels = []

    model.eval()
    for step, (x, y) in enumerate(loader):
        model.zero_grad()

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss_epoch += loss.item()
        predicted_labels.extend(predicted.cpu().detach().numpy())
        true_labels.extend(y.cpu().detach().numpy())

    return loss_epoch, accuracy_epoch, predicted_labels, true_labels


def make_confusion_matrix(label_predicted, label_test, num_classes):
    acc = 0
    confusion = np.zeros((num_classes, num_classes))
    num_samples = len(label_predicted)

    for i, l_pred in enumerate(label_predicted):
        confusion[l_pred, label_test[i]] = confusion[l_pred, label_test[i]] + 1
        if l_pred == label_test[i]:
            acc = acc + 1
        accuracy = acc / num_samples

    for i in range(num_classes):
        confusion[:, i] = confusion[:, i] / np.sum(confusion[:, i])

    return confusion, accuracy


def visualize_confusion_matrix(confusion, accuracy, label_classes, name):

    # There was an issue where inline (iPython) plot settings
    # were different than terminal plot settings. This
    # should force them to be the same:
    plt.rcParams['figure.figsize']=[6.4, 4.8]
    plt.rcParams['figure.dpi']=100
    plt.rcParams['savefig.dpi']='figure'
    plt.rcParams['figure.subplot.bottom']=.11
    plt.rcParams['figure.edgecolor']= 'white'
    plt.rcParams['figure.facecolor']= 'white'

    plt.title("{}, acc = {:.3f}".format(name, accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.savefig(name+'.png')
    # plt.show()
    plt.close()


def lin_eval(train_X, train_y, test_X, test_y, name, classes, DEVICE):

    log_dict = {'train_loss_per_epoch': [],
                'train_accuracy_per_epoch': [],
                'final_loss': [],
                'final_accuracy': [],
                'predicted_labels': [],
                'true_label': []}

    num_classes = len(classes)

    model = LogisticRegression(np.shape(train_X)[1], num_classes)
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
        train_X, train_y, test_X, test_y, batch_size=128)

    logistic_epochs = 500
    for epoch in range(logistic_epochs):
        loss_epoch, accuracy_epoch = train_logistic_regression(
            arr_train_loader, model, criterion, optimizer, DEVICE)
        print(
            f"Epoch [{epoch}/{logistic_epochs}]\t Loss: {loss_epoch / len(arr_train_loader)}\t Accuracy: {accuracy_epoch / len(arr_train_loader)}"
        )
        log_dict['train_loss_per_epoch'].append(loss_epoch / len(arr_train_loader))
        log_dict['train_accuracy_per_epoch'].append(accuracy_epoch / len(arr_train_loader))


    # final testing
    loss_epoch, accuracy_epoch, predicted_labels, true_labels = test_logistic_regression(
        arr_test_loader, model, criterion, optimizer, DEVICE)

    log_dict['predicted_labels'].append(predicted_labels)
    log_dict['true_label'].append(true_labels)

    confusion_name = f'{name}'
    confusion_matrix, accuracy = make_confusion_matrix(predicted_labels, true_labels, num_classes)
    visualize_confusion_matrix(confusion_matrix, accuracy, classes, confusion_name)

    print(f"[FINAL]\t Loss: {loss_epoch / len(arr_test_loader)}\t Accuracy: {accuracy_epoch / len(arr_test_loader)}")

    log_dict['final_loss'].append(loss_epoch/len(arr_test_loader))
    log_dict['final_accuracy'].append(accuracy_epoch/len(arr_test_loader))

    return log_dict


def visualize_tsne(model_name, tsne_xtest, classes, test_y, close_fig=True):

    # For sizing consistancy:
    # plt.figure(figsize=(3.31,5.15))
    plt.rcParams["figure.figsize"] = [5.15, 3.31]

    color = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
    fig, ax = plt.subplots()
    for class_num in np.unique(test_y):
        indices = np.argwhere(test_y==class_num)
        tsne_x_pts = []
        tsne_y_pts = []
        for index in indices:
            tsne_x_pts.append(tsne_xtest[index,0])
            tsne_y_pts.append(tsne_xtest[index,1])
        ax.scatter(tsne_x_pts, tsne_y_pts, color=color[class_num], label=classes[class_num], s=5)
        # ax.scatter(tsne_xtest[:,0], tsne_xtest[:,1], c=test_y, label=classes, s=5)
        # ax.legend()
        legend = ax.legend(loc="lower left", title="Classes")
    # ax.add_artist(legend1)
    ax.grid(True)
    plt.savefig(model_name+'_TSNE.png')
    # plt.show()
    if close_fig:
        plt.close()
    else:
         plt.show()


def visualize_hover_images(model_name, embeddings, test_images, pred_classes, class_names=None, true_classes=None, showplot=False):
    # Converts the images in 8-bit rgb to pillow images to allow embedding into
    # the interactive plot.
    def embeddable_image(data):
        data = np.transpose((255 * data).astype(np.uint8), (1,2,0)) # This line is needed because the img went through the pytorch test_image transform initially
        image = Image.fromarray(data, mode='RGB').resize((64, 64), Image.Resampling.BICUBIC)
        buffer = BytesIO()
        image.save(buffer, format='png')
        for_encoding = buffer.getvalue()
        return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()

    output_file(model_name+'.html')

    if class_names is not None:
        named_classes = []
        for i in pred_classes:
            named_classes.append(class_names[i])
        pred_classes = named_classes

        if true_classes is not None:
            named_true_classes = []
            for i in true_classes:
                named_true_classes.append(class_names[i])
            true_classes = named_true_classes

    if true_classes is None:
        true_classes = np.full_like(pred_classes, 'N/A', dtype=object)

    digits_df = pd.DataFrame(embeddings, columns=('x', 'y'))
    digits_df['pred_classes'] = [str(x) for x in pred_classes]
    digits_df['true_classes'] = [str(x) for x in true_classes]

    digits_df['image'] = list(map(embeddable_image, test_images))
    datasource = ColumnDataSource(digits_df)

    # Setting the colors to span whole range of a palette:
    factors=[str(x) for x in np.unique(pred_classes)]
    if len(factors) < 6:
        palette = Spectral5
    elif len(factors) < 11:
        palette = Spectral10
    else:
        palette = Turbo256

    color_mapping = CategoricalColorMapper(factors=factors, palette=palette)

    plot_figure = figure(
        title="TSNE Projection of "+model_name,
        width=900,
        height=600,
        tools=('pan, wheel_zoom, reset')
    )

    plot_figure.add_tools(HoverTool(tooltips="""
    <div>
        <div>
            <img
                src="@image" height="64" alt="@imgs" width="64"
                style="float: left; margin: 0px 40px 15px 0px"
                border="2"
            ></img>
        </div>
        <div>
            <span style="font-size: 17px; font-weight: bold;">@pred_classes</span>
        </div>
        <div>
        </div>
        <div>
            <span style="font-size: 10px;">Ground Truth:</span>
            <span>@true_classes</span>
        </div>
    </div>
    """))

    plot_figure.circle(
        'x',
        'y',
        source=datasource,
        color=dict(field='pred_classes', transform=color_mapping),
        line_alpha=0.6,
        fill_alpha=0.6,
        size=5,
        legend_field='pred_classes'
    )
    plot_figure.legend.title='Predicted Class:'
    plot_figure.legend.title_text_font_size='12px'
    plot_figure.legend.title_text_font_style='bold'


    # plot_figure.legend.click_policy='mute'

    save(plot_figure)

    if showplot:
        show(plot_figure)


def kmeans_classifier_2class(test_X, test_y):

    if len(np.unique(test_y))>2:
        print("This function is only set up to handle 2 classes.")
        return

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=0).fit(test_X)
    pred_labels1 = kmeans.labels_
    pred_labels2 = ~pred_labels1 + 10 # Swap the labels to check if they're switched

    num_correct = 0
    for arg in pred_labels1 - test_y:
        if arg == 0:
            num_correct+=1
    accuracy1 = num_correct / len(test_y)

    # if the labels are switched:
    num_correct = 0
    for arg in pred_labels2 * test_y:
        if arg == 0:
            num_correct+=1
    accuracy2 = num_correct / len(test_y)

    print('KNN Accuracy: ', max(accuracy1, accuracy2))

    if accuracy1 > accuracy2:
        return pred_labels1
    else:
        return pred_labels2


def kmeans_classifier(test_X, k=10):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, random_state=0).fit(test_X)
    pred_labels = kmeans.labels_
    return pred_labels


def knn_classifier(train_X, train_y, test_X, test_y, k=100):
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train_X, train_y)
    pred_labels = neigh.predict(test_X)

    total_top1 = (pred_labels == test_y).sum().item()
    acc = total_top1 / len(pred_labels)

    print(f"KNN Accuracy (k={k}): ", acc)

    return pred_labels


def linear_classifier(train_X, train_y, test_X, test_y):
    from sklearn.linear_model import LogisticRegression
    LogReg = LogisticRegression(random_state=0)
    LogReg.fit(train_X, train_y)
    pred_labels = LogReg.predict(test_X)

    total_top1 = (pred_labels == test_y).sum().item()
    acc = total_top1 / len(pred_labels)

    print("Logistic Regression Accuracy: ", acc)

    return pred_labels
