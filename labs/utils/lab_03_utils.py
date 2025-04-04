#########################################################################################################
################################### This are functions for lab 03.   ####################################
###################################          DO NOT MODIFY!          ####################################
#########################################################################################################

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable
from sklearn.metrics import f1_score
from typing import List
from sklearn.metrics import accuracy_score, f1_score
import gdown
import zipfile

################################################# PART 1 #################################################

def show_figure(path:str,title:str,figsize:tuple):
    # Check if folder and image exist
    assert os.path.exists(os.path.join(os.getcwd(),path)), "Image not found, please check directory structure"

    # Load image
    
    img = np.array(Image.open(os.path.join(os.getcwd(),path)))
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")  # Cacher les axes
    plt.show() 

def load_data(path: str):
    dataroot = "../data/data_lab_03/part_01"

    set = path.split("_")[1].split(".")[0]

    # train features and labels
    train_data = torch.load(os.path.join(dataroot, path))
    train_x, train_y = train_data["features"], train_data["labels"]

    print(f"Distribution of data in {set} set")
    print("#Tumor examples: {}".format(len(train_y[train_y == 0])))
    print("#Stroma examples: {}".format(len(train_y[train_y == 1])))
    if len(train_y[train_y == -1])>0:
        print("#OoD examples: {}".format(len(train_y[train_y == -1])))

    return train_x, train_y


def mahalanobis_classifier(
    MahalanobisClassifier: Callable, train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, val_y: torch.Tensor, cls_name: List[str], colors: List[str]):
    """Fit the classifier and display Mahalanobis distances for the first two features over samples

    Args:
        fa (torch.Tensor): (N,) First feature component
        fb (torch.Tensor): (N,) Second feature component
        y (torch.Tensor): (N,) Class ground truth
        cls_name (list of str): (n_classes,) Name of classes as list
        title (str): Title of plot
    """
    
    # Create classifier
    classifier = MahalanobisClassifier()

    # Fit to train data
    classifier.fit(train_x, train_y)

    # Apply on validation data and compute accuracy
    val_y_hat, val_y_dist = classifier.predict(val_x)
    accuracy = accuracy_score(val_y, val_y_hat)

    fa=val_y_dist[:,0]
    fb=val_y_dist[:,1]

    # Create plot
    _, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Plot results
    for i, c in enumerate(np.unique(val_y)):
        ax.scatter(fa[val_y == c], fb[val_y == c], marker="o", c=colors[i], label="{}".format(cls_name[i]))
    
    # Labels
    ax.set_xlabel("Distance to Tumor")
    ax.set_ylabel("Distance to Stroma")
    plt.title("Mahalanobis distances for samples\nValidation set accuracy: {:.2f}%".format(100*accuracy))
    plt.legend()
    


def mahalanobis_ood_classifier(
    MahalanobisOODClassifier: Callable, train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, val_y: torch.Tensor, cls_name: List[str], colors: List[str]):
    """Display Mahalanobis distances for the first two features over samples as well as OoD scores.

    Args:
        fa (torch.Tensor): (N,) First feature component
        fb (torch.Tensor): (N,) Second feature component
        ood_score (torch.Tensor): (N,) OoDness of samples
        cls_name (list of str): (n_classes,) Name of classes as list
        title (str): Title of plot
    """
    
    classifier_ood = MahalanobisOODClassifier()

    # Fit to train data
    classifier_ood.fit(train_x, train_y)

    # Apply on validation data and compute accuracy
    val_y_hat, val_y_dist, val_y_ood_scores = classifier_ood.predict(val_x)
    accuracy = accuracy_score(val_y, val_y_hat)

    fa=val_y_dist[:,0]
    fb=val_y_dist[:,1]

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Plot results
    pcm = ax.scatter(fa, fb, c=val_y_ood_scores, marker="o", label="OoD score")
    
    # Labels
    ax.set_xlabel("Distance to Tumor")
    ax.set_ylabel("Distance to Stroma")
    fig.colorbar(pcm, ax=ax, label="OoD score")
    plt.title("OoD scores for samples\nValidation set accuracy: {:.2f}%".format(100*accuracy))
    plt.legend()

    return classifier_ood, val_y_ood_scores

def check_threshold(get_ood_threshold:Callable,val_y_ood_scores:float):
    q = 0.95
    threshold_val = get_ood_threshold(ood_scores=val_y_ood_scores, quantile=q)


    # Plot ood scores and threshold
    plot_ood_scores(ood_scores=val_y_ood_scores, threshold=threshold_val)

    return threshold_val


def plot_ood_scores(ood_scores: torch.Tensor, threshold: float):
    """ Plot OoD scores and the threshold to quantile.

    Args:
        ood_scores (torch.Tensor): (N, ) N measured OoDness scores
        threshold (float): OoD threshold
    """
    print("Validation threshold {:.0f}% = {:.2f}".format(100*0.95, threshold))
    plt.figure(figsize=(12, 4))
    plt.hist(ood_scores[ood_scores <= threshold], bins=20, label="ID")
    plt.hist(ood_scores[ood_scores > threshold], bins=20, label="OoD")
    plt.vlines(threshold, ymin=0, ymax=10, color='k', ls='--', label="Threshold")
    plt.xlabel("OoD scores")
    plt.ylabel("Score density")
    plt.title("OoD scores and threshold")
    plt.legend()
    

def plot_mahalanobis_classifier(
    fa: torch.Tensor, fb: torch.Tensor, y: torch.Tensor, cls_name: List[str], colors: List[str], title: str):
    """Display Mahalanobis distances for the first two features over samples

    Args:
        fa (torch.Tensor): (N,) First feature component
        fb (torch.Tensor): (N,) Second feature component
        y (torch.Tensor): (N,) Class ground truth
        cls_name (list of str): (n_classes,) Name of classes as list
        title (str): Title of plot
    """
    
    # Create plot
    _, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Plot results
    for i, c in enumerate(np.unique(y)):
        ax.scatter(fa[y == c], fb[y == c], marker="o", c=colors[i], label="{}".format(cls_name[i]))
    
    # Labels
    ax.set_xlabel("Distance to Tumor")
    ax.set_ylabel("Distance to Stroma")
    plt.title(title)
    plt.legend()

    

def eval_test(classifier_ood: Callable,compute_metrics:Callable, test_x: torch.Tensor,test_y: torch.Tensor,threshold_val: float):

    test_y_hat, test_y_dist, test_y_ood_scores = classifier_ood.predict(test_x)

    # Compute metrics
    recall_tumor, recall_stroma, recall_ood, avg_recall = compute_metrics(
        y=test_y, y_hat=test_y_hat, ood_scores=test_y_ood_scores, threshold=threshold_val)

    # Display metrics
    print(f"Tumor recall: {recall_tumor*100:.2f}%")
    print(f"Stroma recall: {recall_stroma*100:.2f}%")
    print(f"OoD recall: {recall_ood*100:.2f}%")
    print(f"Average recall: {avg_recall*100:.2f}%")

    # Display the classification with OoD and features
    plot_mahalanobis_classifier(
        fa=test_y_dist[:,0] , fb=test_y_dist[:,1], y=test_y,
        cls_name=["OoD", "Tumor", "Stroma"], colors=["k", "r", "b"],
        title="Mahalanobis distances for samples (Ground Truth)"
    )

    # Display the classification with OoD and features
    plot_mahalanobis_classifier(
        fa=test_y_dist[:,0] , fb=test_y_dist[:,1], y=test_y_hat,
        cls_name=["OoD", "Tumor", "Stroma"], colors=["k", "r", "b"],
        title="Mahalanobis distances for samples (Prediciton)"
    )

    return test_y_dist, test_y_hat
    

def check_best_k(find_best_k: Callable, kNNClassifier: Callable,train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, val_y: torch.Tensor):

    ks = [1, 3, 5, 9, 15, 25]
    # Print best K
    best_k, best_accuracy = find_best_k(ks, kNNClassifier, train_x, train_y, val_x, val_y)
    print(f"\nBest @ k: {best_k} -> {best_accuracy*100:.2f}% accuracy")
    return best_k, best_accuracy

def eval_test_knn(classifier_knn: Callable,compute_metrics:Callable, test_x: torch.Tensor,test_y: torch.Tensor,threshold_val: float):

    try:
        test_y_hat, test_y_ood_scores = classifier_knn.predict(test_x)
        # Compute metrics
        recall_tumor, recall_stroma, recall_ood, avg_recall = compute_metrics(
            y=test_y, y_hat=test_y_hat, ood_scores=test_y_ood_scores, threshold=threshold_val)

        # Display metrics
        print(f"Tumor recall: {recall_tumor*100:.2f}%")
        print(f"Stroma recall: {recall_stroma*100:.2f}%")
        print(f"OoD recall: {recall_ood*100:.2f}%")
        print(f"Average recall: {avg_recall*100:.2f}%")

    except Exception as e:
        print(f"An error occurred: {e}")


################################################# PART 2 #################################################

def show_2_figures(path1:str,title1:str,path2:str,title2:str,figsize:tuple):
    # Check if folder and image exist
    assert os.path.exists(os.path.join(os.getcwd(),path1)), "Image not found, please check directory structure"
    assert os.path.exists(os.path.join(os.getcwd(),path2)), "Image not found, please check directory structure"

    # Load image
    
    img1 = np.array(Image.open(os.path.join(os.getcwd(),path1)))
    img2 = np.array(Image.open(os.path.join(os.getcwd(),path2)))

    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=figsize)
    ax1.imshow(img1)
    ax1.set_title(title1)
    ax1.axis("off")  # Cacher les axes
    ax2.imshow(img2)
    ax2.set_title(title2)
    ax2.axis("off")  # Cacher les axes
    plt.show()
    return 

def download_data():
    if not os.path.exists("../data/data_lab_03/part_02/"):
        url = "https://drive.google.com/uc?id=17xWybfPMJDtC-vZYvsWuwZU1CULPSThq&confirm=t"
        zip_path = "../data/data_lab_03.zip"  # Change the file name as needed
        try:
            gdown.download(url, zip_path, quiet=False, fuzzy=True)

            os.makedirs("../data/", exist_ok=True)

            # Extract the ZIP file
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall("../data/")
        except Exception as e:
            print(f"An error occurred: -- Please download the data manually from {url} and unzip it in the correct folder")


def load_data_2(DHMC2Cls, path):
    dataroot = "../data/data_lab_03/part_02"
    if path.split('_')[1].split('.')[0] == 'train':
        train = True 
    else :
        train = False
    train_dataset = DHMC2Cls(os.path.join(dataroot, path), train=train)
    return train_dataset

def create_dataset(DHMC2Cls:Callable):
    """ Automatic check of implementation, DO NOT Modify

    Args:
        train_dataset (Dataset): Training dataset
        val_dataset (Dataset): Validation dataset

    Return:
        status (str): Return "Successful" of all tests passed, "Failed" otherwise
    """
    # Run the block to build the train and validation datasets
    try:
        train_dataset = load_data_2(DHMC2Cls,  "dhmc_train.pth")
        val_dataset = load_data_2(DHMC2Cls,  "dhmc_val.pth")

        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None
    try:
        # Perform sanity check of the training dataset creation
        assert len(train_dataset) == 59
        features, label = train_dataset[1]
        assert label == 1
        assert np.isclose(features[0, 0], 0.0538, rtol=1e-3)
        
        # Perform sanity check of the validation dataset creation
        assert len(val_dataset) == 40
        features, label, wsi_id, coord = val_dataset[1]
        assert label == 1
        assert np.isclose(features[0, 0], 0.0588, rtol=1e-3)
        assert wsi_id == 'DHMC_0008'
        assert coord[0, 0] == 21697
        
    except Exception:
        print("Failed :(")

    print("Successful :)")
    return train_loader, val_loader


def sanity_check_avg(avg_func: Callable):
    """ Automatic check of implementation, DO NOT Modify

    Args:
        avg_func (Dataset): Average pooling function

    Return:
        status (str): Return "Successful" of all tests passed, "Failed" otherwise
    """
    try:
        # Sanity check for average function
        x_in = torch.Tensor([[0, 3], [4, 6], [2, 0]])
        x_out = torch.Tensor([[2, 3]])
        assert np.all(np.isclose(avg_func().forward(features=x_in), x_out, rtol=1e-3))
        
    except Exception:
        return "Failed :("

    return "Successful :)"

def sanity_check_cls(cls_class: Callable, Pooling: Callable):
    """ Automatic check of implementation, DO NOT Modify

    Args:
        avg_func (nn.Module): Linear classifier

    Return:
        status (str): Return "Successful" of all tests passed, "Failed" otherwise
    """
    try:
        # Test of implementation
        d, H, n_classes = 768, 512, 2
        cls = cls_class(in_dim=d, H=H, n_classes=n_classes, pooling_fn=Pooling())
        assert cls(torch.zeros((1, 1000, d))).shape == torch.Size([1, n_classes])
        
    except Exception:
        return "Failed :("

    return "Successful :)"


@torch.no_grad()
def test(model : nn.Module, test_loader : DataLoader):
    """The test function, computes the F1 score of the current model on the test_loader

    Args:
        model (nn.Module): The model to evaluate
        test_loader (DataLoader): The test data loader to iterate on the dataset to test

    Returns:
        f1 (float): The F1 score on the given dataset
        loss (float): Averaged loss on the given dataset
    """
    try:
        model.eval()

        preds_dict = {"preds" : torch.Tensor(), "labels" : torch.Tensor(), 'losses': torch.Tensor()}
        for features, labels, _, _ in test_loader:
            # Forward and loss
            preds = model(features)
            loss = F.cross_entropy(preds, labels)
            
            # Store values
            preds_dict["preds"] = torch.cat([preds_dict["preds"], preds.argmax(1)])
            preds_dict["labels"] = torch.cat([preds_dict["labels"], labels])
            preds_dict["losses"] = torch.cat([preds_dict["losses"], loss[None]])

        # Compute metric and loss
        f1 = f1_score(preds_dict["labels"], preds_dict["preds"], average="macro")
        loss = preds_dict["losses"].mean()

        print(f"Test F1 score: {100*f1:.2f}%")
    except Exception as e:
        print(f"An error occurred: {e}")



def plot_training(model, train: Callable, train_loader, val_loader, epochs, optimizer):

    """Plot training results of linear classifier
    
    Args:
        best_epoch (int): Best epoch
        val_accs (List): (E,) list of validation measures for each epoch
        val_loss (List): (E,) List of validation losses for each epoch
        train_loss (List): (E,) List of training losses for each epoch
    """
    # Run training and display results
    try:
        best_model, best_f1, best_epoch, val_accs, val_loss, train_loss = train(model, train_loader, val_loader, n_epochs=epochs, optimizer=optimizer)
        print(f"Best model at epoch {best_epoch} -> {100*best_f1:.2f}% F1 score")

        # Create plot
        _, axes = plt.subplots(1, 2, figsize=(12, 4))
        es = np.arange(1, len(val_accs)+1)
        # Plot F1 score
        axes[0].plot(es, val_accs, label="Val")
        axes[0].vlines(best_epoch, ymin=np.min(val_accs), ymax=np.max(val_accs), color='k', ls='--', label="Best epoch")
        axes[0].set_xlabel("Training steps")
        axes[0].set_ylabel("F1-score")
        axes[0].set_title("F1-score")
        axes[0].legend()

        # Plot losses
        axes[1].plot(es, val_loss, label="Val")
        axes[1].plot(es, train_loss, label="Train")
        axes[1].vlines(best_epoch, ymin=np.min(train_loss), ymax=np.max(val_loss), color='k', ls='--', label="Best epoch")
        axes[1].set_xlabel("Training steps")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Losses")
        axes[1].legend()
        
        plt.tight_layout()

        return best_model
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def sanity_gated(func_gated: Callable):
    """ Automatic check of implementation, DO NOT Modify

    Args:
        avg_func (nn.Module): Attention gated

    Return:
        status (str): Return "Successful" of all tests passed, "Failed" otherwise
    """
    try:
        # Test of implementation
        L, M = 512, 256
        layer = func_gated(L, M)
        # Check output size
        xin = torch.zeros(1000, L)
        assert layer(xin).size() == torch.Size([1000, 1])
    except Exception:
        return "Failed :("

    return "Successful :)"

def build_prediction_map(
        coords_x: np.ndarray,
        coords_y:  np.ndarray,
        feature:  np.ndarray,
        wsi_dim: Optional[tuple] = None,
        default: Optional[float] = -1.,
):
    """
    Build a prediction map based on x and y coordinates and feature vectors. Default values if feature is nonexisting
    for a certain location is -1.

    Parameters
    ----------
    coords_x: np.ndarray of shape (N,)
        Coordinates of x points.
    coords_y: np.ndarray of shape (N,)
        Coordinates of y points.
    feature: np.ndarray of shape (N, M)
        Feature vector.
    wsi_dim: tuple of int, optional
        Size of the original whole slide.
    default: float, optional
        Value of the pixel when the feature is not defined.

    Returns
    -------
    map: np.ndarray (W, H, M)
        Feature map. The unaffected points use the default value -1.
    map_x, map_y: np.ndarray (W, H), np.ndarray (W, H)
        Corresponding and y coordinates of the feature map.
    """
    # Compute offset of coordinates in pixel (patch intervals)
    interval_x = np.min(np.unique(coords_x)[1:] - np.unique(coords_x)[:-1])
    interval_y = np.min(np.unique(coords_y)[1:] - np.unique(coords_y)[:-1])

    # Define new coordinates
    offset_x = np.min(coords_x) % interval_x
    offset_y = np.min(coords_y) % interval_y
        
    coords_x_ = ((coords_x - offset_x) / interval_x).astype(int)
    coords_y_ = ((coords_y - offset_y) / interval_y).astype(int)

    # Define size of the feature map
    map = default * np.ones((int(wsi_dim[1] / interval_y), int(wsi_dim[0] / interval_x), feature.shape[1]))
    map[coords_y_, coords_x_] = feature
    
    return map


@torch.no_grad()
def plot_attention(model, test_loader):
    """ Plot attention on top of slide images

    Args:
        model (nn.Module): Model 
        test_loader (Dataloader): Data loader for the test set
    """
    try:
        dataroot = "../data/data_lab_03/part_02"
        # Define new plot
        fig, ax = plt.subplots(2, 2, figsize=(16, 10), height_ratios=[3, 2], width_ratios=[1, 1.25])

        # iterate over slides
        for i, (features, _, wsi_id, coordinates) in enumerate(test_loader):

            # Get data and paths
            wsi_id = wsi_id[0]
            slide_path = os.path.join(dataroot, f"{wsi_id}.jpg")
            # Forward path
            attention = model.pool(model.proj(features.squeeze()), attention_only=True).squeeze()

            # Get WSI dim (Hardcoded)
            if wsi_id == "DHMC_0001":
                label = "Solid"
                wsi_dim= (39839, 30468)
            elif wsi_id == "DHMC_0007":
                label = "Acinar"
                wsi_dim = (47808, 22631)
            else:
                raise NotImplementedError("There is a problem !")

            # Plot results
            slide_im = np.array(Image.open(slide_path))
            ax[i][0].imshow(slide_im)
            ax[i][0].set_title(label)
            ax[i][0].axis('off')
            
            # Show prediction overlay
            prob_map = build_prediction_map(
                    coords_x=coordinates[0,:, 0].numpy(),
                    coords_y=coordinates[0,:, 1].numpy(),
                    feature=attention[:, None],
                    wsi_dim=wsi_dim,
                    default=0,
            )[:, :, 0]

            # Rescale to ouput map size
            prob_map = F.interpolate(torch.Tensor(prob_map)[None, None], slide_im.shape[:2], mode='bilinear', align_corners=False)[0, 0]

            # Plot prediction map
            ax[i][1].imshow(slide_im)
            pcm = ax[i][1].imshow(prob_map, cmap=matplotlib.colormaps['hot'], vmax=torch.quantile(attention, q=0.99), alpha=0.5)
            ax[i][1].axis('off')
            # Add colorbar
            fig.colorbar(pcm, ax=ax[i][1])
            plt.tight_layout()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None