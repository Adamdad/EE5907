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


class LDA_model:
    def __init__(self, N=3):
        self.N = N

    def fit(self, X, Y):
        clusters = np.unique(Y)
        # within_class scatter matrix
        Sw = np.zeros((X.shape[1], X.shape[1]))
        for i in clusters:
            datai = X[Y == i]
            datai = datai-datai.mean(0)
            Swi = np.matmul(datai.T, datai)
            Sw += Swi
        # between_class scatter matrix
        SB = np.zeros((X.shape[1], X.shape[1]))
        u = X.mean(0)
        for i in clusters:
            Ni = X[Y == i].shape[0]
            ui = X[Y == i].mean(0)  # 某个类别的平均值
            SBi = Ni*np.matmul((ui - u).T, ui - u)
            SB += SBi

        S = np.linalg.inv(Sw)*SB
        eigVals, eigVects = np.linalg.eig(S)  # 求特征值，特征向量
        # eigValInd = np.argsort(eigVals)
        # eigValInd = eigValInd[:(-self.N-1):-1]
        eigVects = np.real(eigVects)
        self.large_eigVect = eigVects[:, :self.N]
        lowdim_X = np.dot(X, self.large_eigVect)

        return lowdim_X

    def transform(self, X):
        lowdim_X = np.dot(X, self.large_eigVect)
        return lowdim_X


def LDA_2(X_train, Y_train, is_selfie):
    H, W = 32, 32
    lda_model = LDA_model(N=2)
    lda_x = lda_model.fit(X_train, Y_train)
    eigVect = lda_model.large_eigVect
    for i in range(2):
        ax = plt.subplot(1, 2, i+1)
        ax.imshow(eigVect[:, i].reshape(H, W))
    plt.savefig('fig/PIE_2d_LDA_eigface.pdf')

    plt.figure()
    plt.scatter(list(lda_x[~is_selfie, 0]), list(
        lda_x[~is_selfie, 1]), label='PIEData')
    plt.scatter(list(lda_x[is_selfie, 0]), list(
        lda_x[is_selfie, 1]), label='My Selfie')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig/PIE_2d_LDA.pdf')


def LDA_3(X_train, Y_train, is_selfie):
    H, W = 32, 32
    lda_model = LDA_model(N=3)
    lda_x = lda_model.fit(X_train, Y_train)
    eigVect = lda_model.large_eigVect
    for i in range(3):
        ax = plt.subplot(1, 3, i+1)
        ax.imshow(eigVect[:, i].reshape(H, W))
    plt.savefig('fig/PIE_3d_LDA_eigface.pdf')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(list(lda_x[~is_selfie, 0]), list(
        lda_x[~is_selfie, 1]), list(lda_x[~is_selfie, 2]), label='PIEData')
    ax.scatter(list(lda_x[is_selfie, 0]), list(lda_x[is_selfie, 1]), list(
        lda_x[is_selfie, 2]), label='My Selfie')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig/PIE_3d_LDA.pdf')


def eval(model, X_test, Y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(Y_test, y_pred)
    return acc


def KNN_cls_exp(X_train, Y_train, X_test, Y_test):
    dims = [2, 3, 9]
    is_selfie = (Y_test == 25)
    X_test_PIE = X_test[~is_selfie]
    Y_test_PIE = Y_test[~is_selfie]
    X_test_selfie = X_test[is_selfie]
    Y_test_selfie = Y_test[is_selfie]
    for dim in dims:
        # Apply PCA
        lda_model = LDA_model(N=dim)
        lda_x = lda_model.fit(X_train, Y_train)
        lda_x_test_pie = lda_model.transform(X_test_PIE)
        lda_x_test_selfie = lda_model.transform(X_test_selfie)
        # Train 1-NN
        knn_model = KNeighborsClassifier(n_neighbors=1)
        knn_model.fit(lda_x, Y_train)
        # Test 1-NN
        acc_pie = eval(knn_model, lda_x_test_pie, Y_test_PIE)
        acc_selfie = eval(knn_model, lda_x_test_selfie, Y_test_selfie)
        print(f'{dim}: PIE Accuracy {acc_pie}, Selfie Accuracy {acc_selfie}')


if __name__ == '__main__':
    dataset = PIEDataSet('./PIE/', './PIE/meta.json')
    num_select = 100
    X_train, Y_train = dataset.load_data()
    X_test, Y_test = dataset.load_data(train=False)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    index = list(range(X_train.shape[0]))
    np.random.shuffle(index)
    index = index[:num_select]
    X_train, Y_train = X_train[index], Y_train[index]
    is_selfie = (Y_train == 25)
    # LDA_2(X_train, Y_train, is_selfie)
    # LDA_3(X_train, Y_train, is_selfie)
    KNN_cls_exp(X_train, Y_train, X_test, Y_test)
