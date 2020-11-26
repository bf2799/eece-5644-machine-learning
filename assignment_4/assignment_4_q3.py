import skimage as ski
from skimage import io
import numpy as np
import sklearn as skl
from sklearn import mixture
import matplotlib.pyplot as plt
import colorsys

MAX_COLOR_VAL = 255.0
HEX_LEN = 6


def generate_normalized_feature_vector(img):
    """
    Normalize an image into a single array of feature vectors (row_num, col_num, R, G, B) where each value lies in [0,1]
    :param img: Image object as array of shape (rows, cols, 3) where 3 is RGB values 0-255
    :return: Array of feature vectors, shape (num_pixels, 5)
    """
    max_row = float(img.shape[0]) - 1
    max_col = float(img.shape[1]) - 1
    new_img = np.pad(img, 1)
    # feature_vectors = np.zeros([img.shape[0] * img.shape[1], 3])
    # for row in range(img.shape[0]):
    #     for col in range(img.shape[1]):
    #         avg = np.mean(np.reshape(new_img[row:row + 3, col:col + 3, 1:4], (9, 3)), axis=0)
    #         feature_vectors[row * img.shape[1] + col] = np.r_[np.subtract(img[row][col], avg)]
    feature_vectors = np.zeros([img.shape[0] * img.shape[1], 5])
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            feature_vectors[row * img.shape[1] + col] = np.r_[row / max_row, col / max_col,
                                                              np.divide(img[row][col], MAX_COLOR_VAL)]
    return feature_vectors, img.shape[0], img.shape[1]


# Create normalized feature vector from image
plane_img = ski.io.imread('3096_color.jpg')
bird_img = ski.io.imread('42049_color.jpg')
mural_img = ski.io.imread('us_map.jpg')
plane_feature_vec, rows, cols = generate_normalized_feature_vector(mural_img)
N = 4
plane_gmm = skl.mixture.GaussianMixture(n_components=N, covariance_type='full', tol=1e-4, reg_covar=1e-10, max_iter=500,
                                        n_init=1, init_params='kmeans', verbose=1)
plane_labels = plane_gmm.fit_predict(plane_feature_vec)
hsvs = [(x * 1.0 / N, 1.0, 1.0) for x in range(N)]
rgbs = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsvs))
plane_img_flat = np.array([rgbs[label] for label in plane_labels])
plane_img_flat = np.reshape(plane_img_flat, (rows, cols, 3))
plt.imshow(plane_img_flat)
plt.show()
