from __future__ import division
from cv2 import DECOMP_SVD, imread
import numpy as np
from numpy.lib import real
from scipy import linalg
from data import PIEDataSet
import matplotlib.pylab as plt
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# import seaborn as sns
# sns.set_theme()


class PCA_model:
    def __init__(self, N=3):
        self.N = N

    def fit(self, X):
        X_mean = np.mean(X, axis=0)
        data = X - X_mean
        cov = np.cov(data.T)
        eigVal, eigVect = linalg.eig(cov)
        eigVect = np.real(eigVect)
        self.large_eigVect = eigVect[:, :self.N]
        lowdim_X = data.dot(self.large_eigVect)
        return lowdim_X

    def transform(self, X):
        X_mean = np.mean(X, axis=0)
        data = X - X_mean
        lowdim_X = data.dot(self.large_eigVect)
        return lowdim_X


def PCA_2(X_train, Y_train, is_selfie):
    H, W = 32,32
    pca_model = PCA_model(N=2)
    pca_x = pca_model.fit(X_train)
    eigVect = pca_model.large_eigVect
    for i in range(2):
        ax = plt.subplot(1, 2, i+1)
        ax.imshow(eigVect[:, i].reshape(H, W))
    plt.savefig('fig/PIE_2d_PCA_eigface.pdf')

    plt.figure()
    plt.scatter(list(pca_x[~is_selfie, 0]), list(
        pca_x[~is_selfie, 1]), label='PIEData')
    plt.scatter(list(pca_x[is_selfie, 0]), list(
        pca_x[is_selfie, 1]), label='My Selfie')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig/PIE_2d_PCA.pdf')


def PCA_3(X_train, Y_train, is_selfie):
    H, W = 32,32
    pca_model = PCA_model(N=3)
    pca_x = pca_model.fit(X_train)
    eigVect = pca_model.large_eigVect
    for i in range(3):
        ax = plt.subplot(1, 3, i+1)
        ax.imshow(eigVect[:, i].reshape(H, W))
    plt.savefig('fig/PIE_3d_PCA_eigface.pdf')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(list(pca_x[~is_selfie, 0]), list(
        pca_x[~is_selfie, 1]), list(pca_x[~is_selfie, 2]), label='PIEData')
    ax.scatter(list(pca_x[is_selfie, 0]), list(pca_x[is_selfie, 1]), list(
        pca_x[is_selfie, 2]), label='My Selfie')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig/PIE_3d_PCA.pdf')


def eval(model, X_test, Y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(Y_test, y_pred)
    return acc

def KNN_cls_exp(X_train, Y_train, X_test, Y_test):
    dims = [40, 80, 200]
    is_selfie = (Y_test == 25)
    X_test_PIE = X_test[~is_selfie]
    Y_test_PIE = Y_test[~is_selfie]
    X_test_selfie = X_test[is_selfie]
    Y_test_selfie = Y_test[is_selfie]
    for dim in dims:
        # Apply PCA
        pca_model = PCA_model(N=dim)
        pca_x = pca_model.fit(X_train)
        pca_x_test_pie = pca_model.transform(X_test_PIE)
        pca_x_test_selfie = pca_model.transform(X_test_selfie)
        # Train 1-NN
        knn_model = KNeighborsClassifier(n_neighbors=1)
        knn_model.fit(pca_x, Y_train)
        # Test 1-NN
        acc_pie = eval(knn_model, pca_x_test_pie, Y_test_PIE)
        acc_selfie = eval(knn_model, pca_x_test_selfie, Y_test_selfie)
        print(f'{dim}: PIE Accuracy {acc_pie}, Selfie Accuracy {acc_selfie}')


if __name__ == '__main__':
    dataset = PIEDataSet('./PIE/', './PIE/meta.json')
    num_select = 500
    X_train, Y_train = dataset.load_data()
    X_test, Y_test = dataset.load_data(train=False)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    index = list(range(X_train.shape[0]))
    np.random.shuffle(index)
    index = index[:num_select]
    X_train, Y_train = X_train[index], Y_train[index]
    is_selfie = (Y_train == 25)
    PCA_2(X_train, Y_train, is_selfie)
    # KNN_cls_exp(X_train, Y_train, X_test, Y_test)
