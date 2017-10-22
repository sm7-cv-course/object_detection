import argparse
import glob
import pickle
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from common import SVM, get_hog
sys.path.insert(0, './../dat')

COMMON_W = 20
COMMON_H = 20

# Construct the argument parser and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--positive", required=True, help="Path to '*.png' images with objects.")
ap.add_argument("-n", "--negative", required=True, help="Path to '*.png' the images with negative class.")
ap.add_argument("-m", "--model", required=True, help="Trained classifier params.")
ap.add_argument("-t", "--table", required=False, help="File with table {[label, features vector]}, list of tuples.")
ap.add_argument("-o", "--pca", required=False, help="File with table containing calculated PCA parameters.")
args = vars(ap.parse_args())

# Paths to images for learning.
path_to_images_positive = args["positive"] + '/*.png'
path_to_images_negative = args["negative"] + '/*.png'


def read_all_images(dirname):
    files = glob.glob(dirname)
    vec_of_images = []
    for file in files:
        img = cv2.imread(file)
        vec_of_images.append(img)
    return vec_of_images


def normalize_images(images):
    images_norm = []
    for img in images:
        if img is not None:
            img_norm = cv2.resize(img, (COMMON_W, COMMON_H), interpolation=cv2.INTER_CUBIC)
            images_norm.append(img_norm)
    return images_norm


def evaluate_model(model, samples, labels):
    """
	Model evaluation.
    """
    resp = model.predict(samples)
    err = (labels != resp).mean()
    print('Accuracy: %.2f %%' % ((1 - err) * 100))

    confusion = np.zeros((2, 2), np.int32)
    for i, j in zip(labels, resp):
        confusion[int(i), int(j)] += 1
    print('confusion matrix:')
    print(confusion)


def pca_projection(pcaset, maxComponents, testset, compressed):
    cv2.PCA(pcaset, cv2.Mat(), cv2.PCA_DATA_AS_ROW, maxComponents)


def compute_hog(hog, imgs):
    """
    Computes HOG features for each image from imgs.
    :param hog: HOG descriptor;
    :param imgs: set of images;
    :return: set of HOGs.
    """
    hog_descriptors = []
    for img in imgs:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        hog_descriptors.append(hog.compute(img_gray))
    return np.squeeze(hog_descriptors)


def compute_HOGs(path_to_images_positive, path_to_images_negative):
	"""
	Computes HOGs for given paths to positive and negative samples.
	:param path_to_images_positive: path to positive sample;
    :param path_to_images_negative: path to negative sample;
    :return: zip(labels, hogs).
	"""
    # If HOGs are precomputed load them
    # Read table {labels, HOG features} from saved in advance file.
    table_path = args["table"]
    if table_path is None:
        table_path = 'hog_descriptors.src'

    print("Load table {labels, features}...")
    try:
	    with open(table_path, 'rb') as f:
		    # List of tuples.
		    table = pickle.load(f)
    except:
        table = None

    if table is not None:
        return table
    else:
        print("Compute HOGs...")
        # Read dataset.
        images_obj = read_all_images(path_to_images_positive)
        images_backs = read_all_images(path_to_images_negative)
        # Normalize images (both train and test sets):
        # 1) bring to common size;
        # 2) histogram equalization - ?
        images_obj_norm = normalize_images(images_obj)
        images_backs_norm = normalize_images(images_backs)

        # Get HOG parameters
        hog = get_hog()

        # Compute features for positives.
        hog_descriptors_obj = compute_hog(hog, images_obj_norm)

        # Compute features for negative sample.
        hog_descriptors_neg = compute_hog(hog, images_backs_norm)

        # Save features to external analysis.
        labels = np.int32(np.hstack((np.ones(len(images_obj_norm)), np.zeros(len(images_backs_norm)))))
        hogs = np.vstack((hog_descriptors_obj, hog_descriptors_neg))

        # Create table as list of tuples.
        table = list(zip(labels, hogs))

        # Write whole list to file.
        with open(table_path, 'wb') as f:
            pickle.dump(table, f)

        return table


# Load HOG or compute if necessary.
table = compute_HOGs(path_to_images_positive, path_to_images_negative)
Labels, Features = zip(*table)
Labels = np.array(Labels)
Features = np.array(Features)

# Divide all features to train and test sets
get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
hog_descriptors_obj = Features[np.array(get_indexes(1, Labels))]
hog_descriptors_backs = Features[get_indexes(0, Labels)]

# Perform PCA on positive sample.
print("Perform PCA...")
mean = np.mean(hog_descriptors_obj, axis=0).reshape(1, -1)
mean, eigenvectors_pca = cv2.PCACompute(hog_descriptors_obj, mean)

# Compute covariance matrix.
# covar, mean = cv2.calcCovarMatrix(hog_descriptors_obj, mean, cv2.COVAR_USE_AVG)# cv2.COVAR_SCRAMBLED)
# covar = np.cov(hog_descriptors_obj.T)
# Covariation matrix computed for all features.
covar = np.cov(Features.T)


# Estimate principal components.
retval, eigenvalues, eigenvectors = cv2.eigen(covar)

# Visualize eigenvalues according to importance.
print("Visualization...")
tot = sum(eigenvalues)
var_exp = [(i / tot) for i in sorted(eigenvalues, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# plt.bar(range(len(var_exp)),
#         var_exp,
#         alpha=0.5,
#         align='center',
#         label='individual explained variance')

plt.step(range(cum_var_exp.shape[0]), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('./figures/pca1.png', dpi=300)
plt.show()

# Apply PCA to datasets
n_main_vals = 143
X_train = np.hstack((hog_descriptors_obj.T, hog_descriptors_backs.T))
X_train_PCA = (X_train.T.dot(eigenvectors[0:n_main_vals].T)).astype(np.float32)

# Train SVM classifier
print('Training SVM model ...')
model = SVM()
model.train(X_train_PCA, Labels)

print('Saving SVM model ...')
model.save(args["model"])

# Test SVM classifier
evaluate_model(model, X_train_PCA, Labels)

print("Save PCA parameters...")
pca_path = args["pca"]
if pca_path is None:
    pca_path = './../dat/pca.src'

table = list(zip(eigenvalues[0:n_main_vals], eigenvectors[0:n_main_vals]))

with open(pca_path, 'wb') as f:
    pickle.dump(table, f)
