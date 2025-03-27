#########################################################################################################
################################### This are functions for lab 02.   ####################################
###################################          DO NOT MODIFY!          ####################################
#########################################################################################################
import os
import wget
import matplotlib.pyplot as plt
import numpy as np

from mnist import MNIST
from typing import Callable


def load_lab02_data():
    # Create a local folder
    folder_lab = os.path.join("..", "data", "data_lab_02")
    os.makedirs(folder_lab, exist_ok=True)
    # Data url
    url_img = "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz"
    url_label = "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz"

    # Data filename 
    file_img = os.path.join(folder_lab, "train-images-idx3-ubyte.gz")
    file_label = os.path.join(folder_lab, "train-labels-idx1-ubyte.gz")

    # Download data 
    if not os.path.exists(file_img):
        file_img = wget.download(url_img, out=file_img)
        file_label = wget.download(url_label, out=file_label)

    print("Data downloaded under folder: {}".format(folder_lab))

    # Load data
    mndata = MNIST(folder_lab, gz=True)
    images, labels = mndata.load_training()

    # Convert as numpy arrays and binarize images
    images = np.array(images).reshape((-1, 28, 28))
    labels = np.array(labels)

    print("{} Images and {} labels loaded".format(len(images), len(labels)))
    display_samples(images[:24], labels[:24], title="First 24 samples with labels")
    return images, labels


def display_samples(images: np.ndarray, labels:np.ndarray, title: str, cnt: list = None):
    """
    Display images along with labels. 
    
    Args
    ----
    images: np.ndarray (N, 28, 28)
        Source images
    labels: np.ndarray (N)
        List of labels associated with the input image
    title: str
        Title of the plot
    cnt: list
        List of contours to display (only used for exercise 1.3 and more)
    """

    # Get the number of images, columns, and rows
    n = len(images)
    n_cols = 8
    r_rows = np.ceil(n/n_cols).astype(int)
    
    # Define plot
    _, axes = plt.subplots(r_rows, n_cols, figsize=(14, 2*r_rows))
    axes = axes.ravel()
    
    
    # Plot all images and labels
    for i in range(n):
        axes[i].imshow(images[i], interpolation="nearest")
        axes[i].axis("off")
        axes[i].set_title(labels[i])
        # Check if need to display contour
        if cnt is not None and len(cnt) == n:
            axes[i].plot(cnt[i][:, 0], cnt[i][:, 1], 'r-*')

    # Set title
    plt.suptitle(title)
    plt.tight_layout()

def plot_features(features_a: np.ndarray, features_b: np.ndarray, label_a: str, label_b: str, title: str):
    """
    Plot feature components a and b.
    
    Args
    ----
    features_a: np.ndarray (N, D)
        Feature a with N samples and D complex features. 
    features_b: np.ndarray (N, D)
        Feature b with N samples and D complex features.
    label_a: str
        Name of the feature a.
    label_b: str
        Name of the feature b.
    """

    # Number of paris to display
    n_features = features_a.shape[1]
    # Define pairs for 2D plots
    pairs = np.array(range(2*np.ceil(n_features / 2).astype(int)))
    # Check if odd lenght, shift second feature to have pairs
    if n_features % 2 == 1:
        pairs[2:] = pairs[1:-1]
    # Convert to 2d array
    pairs = pairs.reshape(-1, 2)

    # Plot each pairs and labels
    n_plots = len(pairs)
    _, axes = plt.subplots(3, n_plots, figsize=(15, 8))
    
    for i, (pa, pb) in enumerate(pairs):
        # Real
        axes[0, i].scatter(np.real(features_a[:, pa]), np.real(features_a[:, pb]), label=label_a, s=10, alpha=0.1)
        axes[0, i].scatter(np.real(features_b[:, pa]), np.real(features_b[:, pb]), label=label_b, s=10, alpha=0.1)
        axes[0, i].set_xlabel("Component {}".format(pa))
        axes[0, i].set_ylabel("Component {}".format(pb))
        axes[0, i].set_title("Real {} vs {}".format(pa, pb))
        axes[0, i].legend()
        # Imag
        axes[1, i].scatter(np.imag(features_a[:, pa]), np.imag(features_a[:, pb]), label=label_a, s=10, alpha=0.1)
        axes[1, i].scatter(np.imag(features_b[:, pa]), np.imag(features_b[:, pb]), label=label_b, s=10, alpha=0.1)
        axes[1, i].set_xlabel("Component {}".format(pa))
        axes[1, i].set_ylabel("Component {}".format(pb))
        axes[1, i].set_title("Imag. {} vs {}".format(pa, pb))
        axes[1, i].legend()
        # Abs
        axes[2, i].scatter(np.abs(features_a[:, pa]), np.abs(features_a[:, pb]), label=label_a, s=10, alpha=0.1)
        axes[2, i].scatter(np.abs(features_b[:, pa]), np.abs(features_b[:, pb]), label=label_b, s=10, alpha=0.1)
        axes[2, i].set_xlabel("Component {}".format(pa))
        axes[2, i].set_ylabel("Component {}".format(pb))
        axes[2, i].set_title("Abs. {} vs {}".format(pa, pb))
        axes[2, i].legend()

    plt.suptitle(title)
    plt.tight_layout()


def apply_transformation(imgs: np.ndarray, func: Callable):
    """
    Apply random transformation to a set of images

    Args
    ----
    image: np.ndarray (N, 28, 28)
        Source images
    func: Callable
        Transformation function to apply to input images
        
    Return
    ------
    imgs_trans: np.ndarray (N, 28, 28)
        Transformed images
    """

    # Get the number of images
    n = len(imgs)
    imgs_trans = np.zeros_like(imgs)

    # Apply transformation
    for i in range(n):
        imgs_trans[i] = func(imgs[i])

    return imgs_trans
    
def plot_transform(img, function, title):
    """
    Plot random transformation for visualization purposes

    Args
    ----
    image: np.ndarray (28, 28)
        Source image
    func: Callable
        Transformation function to apply to input images
    title: str
        Title of the plot
    """
    
    # Fix the number of examples to display 
    n = 10
    _, axes = plt.subplots(1, n, figsize=(16, 2))

    # Apply n random transformation on input images
    for i in range(n):
        trans = function(img=img)
        axes[i].imshow(trans, interpolation='nearest')
        axes[i].axis("off")

    plt.suptitle(title)
    plt.tight_layout()

def plot_reference_patterns(
    pattern_a: np.ndarray, pattern_b: np.ndarray, title_a: str, title_b: str, map_a: np.ndarray = None, map_b: np.ndarray = None
):
    """
    Plot the reference patterns for the two patterns as well as distance maps if provided

    Args
    ----
    pattern_a: np.ndarray (28, 28)
        The first pattern to display
    pattern_b: np.ndarray (28, 28)
        The second pattern to display 
    title_a: str
        Title of the first plot
    title_a: str
        Title of the first plot 
    map_a: np.ndarray (28, 28)
        Distance map, If None, the map is not plotted
    map_b: np.ndarray (28, 28)
        Distance map 2nd, If None, the map is not plotted
    """
    
    # Display results
    fig, axes = plt.subplots(1, 4, figsize=(10, 2))
    # Remove axes
    [a.axis("off") for a in axes]
    # Show patterns
    axes[0].imshow(pattern_a, interpolation="nearest")
    axes[0].set_title(title_a)
    axes[1].imshow(pattern_b, interpolation="nearest")
    axes[1].set_title(title_b)
    # Check if distance map exists
    if map_a is not None:
        pcm = axes[2].imshow(map_a, interpolation='nearest')
        axes[2].set_title(title_a + "\n(Dist. map)")
        fig.colorbar(pcm, ax=axes[2])
    if map_a is not None:
        pcm = axes[3].imshow(map_b, interpolation='nearest')
        axes[3].set_title(title_b + " \n(Dist. map)")
        fig.colorbar(pcm, ax=axes[3], label="distance to shape")
    plt.tight_layout()

def plot_dmap_features(fa: np.ndarray, fb: np.ndarray, la: str, lb: str):
    """
    Plot distance features for features A and B.

    Args
    ----
    fa: np.ndarray (2, N)
        Features A. Represent distance to self (a->a) and to other (a->b)
    fb: np.ndarray (2, N)
        Features B. Represent distance to self (b->a) and to other (b->b)
    la: str
        Axis label for feature A
    lb: str
        Axis label for feature B
    """
    
    # Define plot
    _, ax = plt.subplots(1, 1, figsize=(5, 5))
    # Plot features and display labels
    ax.scatter(fa[0], fa[1], label="{}".format(la), alpha=0.3)
    ax.scatter(fb[0], fb[1], label="{}".format(lb), alpha=0.3)
    ax.set_xlabel("Distance to pattern {}".format(la))
    ax.set_ylabel("Distance to pattern {}".format(lb))
    plt.legend()

def plot_other_features(
    f_peri: np.ndarray, f_area: np.ndarray, f_comp: np.ndarray, f_rect: np.ndarray, la: str, lb: str):
    """
    Plot features distribution based on input features for the two digits.
    
    Args
    ----
    f_peri: np.ndarray (2, N)
        Estimated perimeter length for both digits
    f_area: np.ndarray (2, N)
        Estimated area for both digits
    f_comp: np.ndarray (2, N)
        Estimated compacity for both digits
    f_rect: np.ndarray (2, N)
        Estimated rectangularity for both digits
    la: str
        Axis label for feature A
    lb: str
        Axis label for feature B
    """

    # Define plot
    _, axes = plt.subplots(1, 2, figsize=(10, 4))

    # peri vs area
    axes[0].scatter(f_peri[0], f_area[0], label=la, alpha=0.3)
    axes[0].scatter(f_peri[1], f_area[1], label=lb, alpha=0.3)
    axes[0].set_xlabel("Perimeter")
    axes[0].set_ylabel("Area")
    axes[0].set_title("Perimeter vs area")
    axes[0].legend()
    
    # compacity vs rectangularity
    axes[1].scatter(f_comp[0], f_rect[0], label=la, alpha=0.3)
    axes[1].scatter(f_comp[1], f_rect[1], label=lb, alpha=0.3)
    axes[1].set_xlabel("Compacity")
    axes[1].set_ylabel("Rectangularity")
    axes[1].set_title("Compacity vs rectangularity")
    axes[1].legend()
    
    plt.tight_layout()
    
    
def test_1_1(extract_label, images, labels):
    """
    Test function for exercise 1.1
    """
    
    # Set the number of digits to include
    n = 1000
    # Set the number of digits to display
    n_plot=24
    
    # Set which digit to consider
    label_a = 0
    label_b = 4

    # Filter images
    images_a = extract_label(images, labels=labels, target_label=label_a)[:n]
    images_b = extract_label(images, labels=labels, target_label=label_b)[:n]

    # Display random results
    display_samples(images=images_a[:n_plot], labels=[str(label_a)]*n_plot, title="Selected {}s (n={})".format(label_a, n_plot))
    display_samples(images=images_b[:n_plot], labels=[str(label_b)]*n_plot, title="Selected {}s (n={})".format(label_b, n_plot))
    
    return images_a, images_b
    
    
def test_1_2(preprocess, images_a, images_b):
    """
    Test function for exercise 1.2
    """
    # Set the number of digits to display
    n_plot=24
    # Set which digit to consider
    label_a = 0
    label_b = 4

    # Extract images with 0s ans 1s
    images_p_a = preprocess(images=images_a)
    images_p_b = preprocess(images=images_b)

    # Display results
    display_samples(images=images_p_a[:n_plot], labels=[label_a]*n_plot, title="Selected {}s (n={})".format(label_a, n_plot))
    display_samples(images=images_p_b[:n_plot], labels=[label_b]*n_plot, title="Selected {}s (n={})".format(label_b, n_plot))
    
    return images_p_a, images_p_b
    
def test_2_1(find_contour, images_p_a, images_p_b):
    """
    Test function for exercise 2.1
    """
    # Set the number of digits to display
    n_plot=24
    # Set which digit to consider
    label_a = 0
    label_b = 4
    
    # Get contours
    cnt_p_a = find_contour(images_p_a)
    cnt_p_b = find_contour(images_p_b)

    print(len(cnt_p_a))

    # Define plot titles
    title_a = "Preprocessed {}s w/ contours (n={})".format(label_a, n_plot)
    title_b = "Preprocessed {}s w/ contours (n={})".format(label_b, n_plot)

    # Display results
    display_samples(
        images=images_p_a[:n_plot], labels=[label_a]*n_plot, cnt=cnt_p_a[:n_plot], title=title_a)
    display_samples(
        images=images_p_b[:n_plot], labels=[label_b]*n_plot, cnt=cnt_p_b[:n_plot], title=title_b)
    
    return cnt_p_a, cnt_p_b

def test_2_1_2(compute_descriptor_padding, cnt_p_a, cnt_p_b):
    """
    Test function for exercise 2.1.2
    """
    n_samples = 11
    # Set which digit to consider
    label_a = 0
    label_b = 4
    # Compute the descriptors based on the contours
    feat_a = compute_descriptor_padding(contours=cnt_p_a, n_samples=n_samples)
    feat_b = compute_descriptor_padding(contours=cnt_p_b, n_samples=n_samples)

    # Plot components
    plot_features(
        features_a=feat_a,
        features_b=feat_b,
        label_a=label_a, label_b=label_b,
        title="Real/Imag./Absolute features",
    )
    
def test_2_1_5(linear_interpolation, cnt_p_a, images_p_a):
    
    '''
    Test function for exercise 2.1.5
    '''
    
    # Get a different number of samples for the contour of the shapes
    n_samples_test = [2, 5, 10, 15, 20, 40, 60, 80]

    # Resample contours
    cs = []
    for i, n in enumerate(n_samples_test):
        c = linear_interpolation(cnt_p_a, n_samples=n)
        cs.append(c[0])

    # Display results with overlay
    display_samples(
        images=np.repeat(images_p_a[0][None], repeats=8, axis=0), 
        labels=["n={}".format(n) for n in n_samples_test], 
        cnt=cs, 
        title="Contour interpolation w/ different values"
    )
    
def test_2_1_6(compute_descriptor_padding, linear_interpolation, cnt_p_a, cnt_p_b, n_samples=11):
    '''
    Test function for exercise 2.1.6
    '''
    
    # Set which digit to consider
    label_a = 0
    label_b = 4
    
    # Compute feature descriptors with resampling
    feat_a = compute_descriptor_padding(contours=linear_interpolation(cnt_p_a, n_samples=n_samples), n_samples=n_samples)
    feat_b = compute_descriptor_padding(contours=linear_interpolation(cnt_p_b, n_samples=n_samples), n_samples=n_samples)

    # Plot components
    plot_features(
        features_a=feat_a,
        features_b=feat_b,
        label_a=label_a, label_b=label_b,
        title="Real/Imag./Absolute features (resampling)",
    )
    
    return feat_a, feat_b

def test_2_3(apply_rotation, apply_scaling, apply_translate, img):
    """
    Test function for exercise 2.3
    """
    # Display random rotations
    plot_transform(img=img, function=apply_rotation, title="Apply different rotation on image")
    # Display random scaling
    plot_transform(img=img, function=apply_scaling, title="Apply different scaling on image")
    # Display random translations
    plot_transform(img=img, function=apply_translate, title="Apply different translation on image")
    

    
def test_2_3_2(translation_invariant, find_contour, apply_translate, compute_descriptor_padding, linear_interpolation,images_p_a, feat_a):
    
    '''
    Test function for exercise 2.3
    '''
    
    # Set which digit to consider
    label_a = 0
    
    # Get descriptors for translated images
    cnt_t_a = find_contour(apply_transformation(imgs=images_p_a, func=apply_translate))
    feat_t_a = compute_descriptor_padding(contours=linear_interpolation(cnt_t_a))

    # Get invariant features
    a = translation_invariant(feat_a)
    b = translation_invariant(feat_t_a)

    # Compute errors
    error_no_corr = np.abs(feat_a - feat_t_a).mean()
    error_corr = np.abs(a - b).mean()
    # Print averaged error before
    print("Translation error: {:.2f}".format(error_no_corr))
    # Print averaged error after
    print("Corrected translation error: {:.2f}".format(error_corr))

    # plot features distribution
    plot_features(
        features_a=feat_a,
        features_b=feat_t_a,
        label_a=label_a, label_b=str(label_a)+"-trans",
        title="Features w/o translation correction (error: {:.2f})".format(error_no_corr),
    )

    plot_features(
        features_a=a,
        features_b=b,
        label_a=label_a, label_b=str(label_a)+"-trans",
        title="Features w/ translation correction (error: {:.2f})".format(error_corr),
    )
    
    return feat_t_a
def test_2_3_3(rotation_invariant, find_contour, apply_rotation, compute_descriptor_padding, linear_interpolation,images_p_a, feat_a):
    
    '''
    Test function for exercise 2.3.3
    '''
    
    # Set which digit to consider
    label_a = 0
    
    # Get descriptors for rotation imagescan
    cnt_r_a = find_contour(apply_transformation(imgs=images_p_a, func=apply_rotation))
    feat_r_a = compute_descriptor_padding(contours=linear_interpolation(cnt_r_a))

    # Get invariant features
    a = rotation_invariant(feat_a)
    b = rotation_invariant(feat_r_a)

    # Compute errors
    error_no_corr = np.abs(feat_a - feat_r_a).mean()
    error_corr = np.abs(a - b).mean()
    # Print averaged error before
    print("Rotation error: {:.2f}".format(error_no_corr))
    # Print averaged error after
    print("Corrected rotation error: {:.2f}".format(error_corr))

    # plot features distribution
    plot_features(
        features_a=feat_a,
        features_b=feat_r_a,
        label_a=label_a, label_b=str(label_a)+"-rot",
        title="Features w/o rotation correction (error: {:.2f})".format(error_no_corr),
    )

    plot_features(
        features_a=a,
        features_b=b,
        label_a=label_a, label_b=str(label_a)+"-rot",
        title="Features w/ rotation correction (error: {:.2f})".format(error_corr),
    )
    

def test_2_3_4(scaling_invariant, find_contour, apply_scaling, compute_descriptor_padding, linear_interpolation,images_p_a, feat_t_a, feat_a):

    '''
    Test function for exercise 2.3.4
    '''
    
    # Set which digit to consider
    label_a = 0
    
    # Get descriptors for scaling images
    cnt_s_a = find_contour(apply_transformation(imgs=images_p_a, func=apply_scaling))
    feat_s_a = compute_descriptor_padding(contours=linear_interpolation(cnt_s_a))

    # Get invariant features
    a = scaling_invariant(feat_a)
    b = scaling_invariant(feat_s_a)

    # Compute errors
    error_no_corr = np.abs(feat_a - feat_t_a).mean()
    error_corr = np.abs(a - b).mean()
    # Print averaged error before
    print("Scaling error: {:.2f}".format(error_no_corr))
    # Print averaged error after
    print("Corrected scaling error: {:.2f}".format(error_corr))

    # plot features distribution
    plot_features(
        features_a=feat_a,
        features_b=feat_s_a,
        label_a=label_a, label_b=str(label_a)+"-sca",
        title="Features w/o scaling correction (error: {:.2f})".format(error_no_corr),
    )

    plot_features(
        features_a=a,
        features_b=b,
        label_a=label_a, label_b=str(label_a)+"-sca",
        title="Features w/ scaling correction (error: {:.2f})".format(error_corr),
    )
    
def test_3_1(reference_pattern, images_p_a, images_p_b):
    
    '''
    Test function for exercise 3.1
    '''
    
    # Set which digit to consider
    label_a = 0
    label_b = 4
    
    # Compute the pattern for both digits
    pattern_a = reference_pattern(images_p_a)
    pattern_b = reference_pattern(images_p_b)

    plot_reference_patterns(
        pattern_a=pattern_a,
        pattern_b=pattern_b, 
        title_a="Reference pattern {}".format(label_a),
        title_b="Reference pattern {}".format(label_b),
    )
    
    return pattern_a, pattern_b


def plot_reconstruction(image, descriptor, compute_reverse_descriptor):
        
    """
    Plot Fourier descriptors reconstruction.
    
    Args
    ----
    image: np.ndarray (28, 28)
        Source images
    descriptor: np.ndarray (D, )
        Complex descriptor with D features
    """
    # Get number of samples
    n_samples = len(descriptor)
    n_mid = n_samples // 2
    # Get intervals
    n_rec = np.linspace(0, n_mid, n_mid+1).astype(int)

    # Plot reconstruction
    _, axes = plt.subplots(1, len(n_rec), figsize=(16, 5))
    
    for i, n in enumerate(n_rec):
        # Create a local copy of the descriptor
        d = descriptor.copy()
        # Remove high frequencies (set to 0)
        d = np.fft.fftshift(d)
        d[:n_mid-n] = 0
        d[n_mid+1+n:] = 0
        # Reverse descriptors to coordinates
        x, y = compute_reverse_descriptor(descriptor=np.fft.ifftshift(d), n_samples=n_samples)
        # Plot contour with image overlay
        axes[i].imshow(image, interpolation='nearest')
        axes[i].scatter(x, y)
        axes[i].plot(x, y)
        axes[i].axis('off')
        axes[i].set_title("N frequencies = {}".format(1 + 2*n))

    plt.tight_layout()


def test_2_2(images_p_a , images_p_b, feat_a, feat_b, compute_reverse_descriptor):
    plot_reconstruction(image=images_p_a[0], descriptor=feat_a[0, :], compute_reverse_descriptor=compute_reverse_descriptor)
    plot_reconstruction(image=images_p_b[0], descriptor=feat_b[0, :], compute_reverse_descriptor=compute_reverse_descriptor)


def test_3_1_2(compute_distance_map, pattern_a, pattern_b):
    
    '''
    Test function for exercise 3.1.2
    '''
    label_a = 0
    label_b = 4
    
    # Compute distance maps
    map_a = compute_distance_map(pattern_a)
    map_b = compute_distance_map(pattern_b)

    plot_reference_patterns(
        pattern_a=pattern_a,
        pattern_b=pattern_b, 
        map_a=map_a,
        map_b=map_b,
        title_a="Reference pattern {}".format(label_a),
        title_b="Reference pattern {}".format(label_b),
    )
    
    return map_a, map_b


def test_3_1_3(compute_distance, images_p_a, images_p_b, map_a, map_b):
    
    '''
    Test function for exercise 3.1.3
    '''
    label_a = 0
    label_b = 4
    
    # Get reference feature a->a, a->b, b->a, and b->b
    d_a2a = compute_distance(images_p_a, map_a)
    d_a2b = compute_distance(images_p_a, map_b)
    d_b2a = compute_distance(images_p_b, map_a)
    d_b2b = compute_distance(images_p_b, map_b)

    # Plot results
    plot_dmap_features(fa=np.stack([d_a2a, d_a2b]), fb=np.stack([d_b2a, d_b2b]), la=label_a, lb=label_b)
    
    
def test_3_2(compute_features, images_p_a, images_p_b):
    
    '''
    Test function for exercise 3.2
    ''' 

    label_a = 0
    label_b = 4
    
    # Get features
    fa_peri, fa_area, fa_comp, fa_rect = compute_features(images_p_a)
    fb_peri, fb_area, fb_comp, fb_rect = compute_features(images_p_b)

    # Plot results
    plot_other_features(
        f_peri=np.stack([fa_peri, fb_peri]),
        f_area=np.stack([fa_area, fb_area]),
        f_comp=np.stack([fa_comp, fb_comp]),
        f_rect=np.stack([fa_rect, fb_rect]),
        la=label_a, lb=label_b
    )

