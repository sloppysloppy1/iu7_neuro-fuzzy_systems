import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.special import softmax
from sklearn import neighbors, datasets

import plotly.express as px
from sklearn.decomposition import PCA

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


from matplotlib.colors import ListedColormap

from prettytable import PrettyTable

le = LabelEncoder()
categories = ['product_title', 'ships_from_to', 'quality', 'btc_price', 'cost_per_gram', 'product_link', 'escrow']

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#DC143C', '#006400'])
colors = {0: 'red', 1: 'green'}
resolution = 1e-4

def test_k(X, y):
    F_measures = []
    t = PrettyTable(['k', 'conf_matrix', '0', '1', 'accuracy', 'R', 'P', 'F_measure'])
    for nbrs in range(2, 26):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        knn = KNeighborsClassifier(n_neighbors=nbrs)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        _confusion_matrix = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = _confusion_matrix.ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        R = tp / (tp + fn)
        P = tp / (tp + fp)
        F_measure = tp / (2 * tp + fp + fn)
        F_measures.append(F_measure)
        t.add_row(['', 0] + list(_confusion_matrix[0]) + [''] * 4)
        t.add_row([nbrs, 1] + list(_confusion_matrix[1]) + [acc, R, P, F_measure])

    print(t)

    plt.plot(range(2, 26), F_measures)
    plt.xlabel('k')
    plt.ylabel('F_measure')
    plt.show()

    return F_measures.index(max(F_measures)) + 2

def plot(X_test, y_test, X_train, y_train, ax, type = 'knn', k_neighbours = 0):
    if type == 'knn':
        clf = KNeighborsClassifier(n_neighbors=k_neighbours)
    else:
        clf = DecisionTreeClassifier()

    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)

    x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()
    y_min, y_max = X_train[:, 1].min(), X_train[:, 1].max()

    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, cmap=cmap_light)
    ax.axis('tight')

    for label in np.unique(y_test):
        indices = np.where(y_test == label)
        ax.scatter(X_test[indices, 0], X_test[indices, 1], c=colors[label], alpha=0.8, cmap=cmap_bold,
                    label='class {}'.format(label))

    print(type, accuracy_score(y_test, y_predict))

df = pd.read_csv('dream_market_cocaine_listings.csv', delimiter=',', encoding ='latin-1')
df = df[categories]

print(df)

for category in categories:
    df[category] = le.fit_transform(df[category])

    if category != 'escrow':
        df[category] = softmax(df[category])

for category in categories:
    print(df[category].describe())

print(df)
#boxplot = df.boxplot()
#plt.show()
# без выбросов
df = df[(df.product_title < 0.06) & (df.btc_price < 0.06) & (df.cost_per_gram < 0.06) & (df.product_link < 0.06)]
print(df)
#boxplot = df.boxplot()
#plt.show()

X = df.iloc[:, [0, 1, 2, 3, 4, 5]].values
y = df.iloc[:, -1].values

#test_k(X,y)

X_train, X_test, y_train, y_test = train_test_split(X, y)

pca_model = PCA(n_components=2)
pca_model.fit(X_train)

X_train = pca_model.transform(X_train)
X_test = pca_model.transform(X_test)

X = pca_model.transform(X)
k_neighbours = test_k(X,y)
print(k_neighbours)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

plot(X_test, y_test, X_train, y_train, ax1, 'knn', k_neighbours)
plot(X_test, y_test, X_train, y_train, ax2, 'tree')

plt.show()
