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


def plot_data(X_train, y_train, clf = 'knn'):
    X = X_train[:, 4:6]
    y = y_train
    h = .02
    if clf == 'knn':
        clf = KNeighborsClassifier(n_neighbors=3)
    else:
        clf = DecisionTreeClassifier()
    clf.fit(X, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    knn = KNeighborsClassifier(n_neighbors=3)
    pca = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=0))
    pca.fit(X_train, y_train)

    knn.fit(pca.transform(X_train), y_train)

    acc_knn = knn.score(pca.transform(X_test), y_test)
    print(acc_knn)
    X_embedded = pca.transform(X)
    X_embedded = X_embedded[:1499]

    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y[:1499], s=30, cmap="Set1")
    plt.show()

    #test_k(X, y)

    plt.figure()
    model.fit(X_train, y_train)

    pca = PCA(n_components=3)
    components = pca.fit_transform(df)
    print(components)
    total_var = pca.explained_variance_ratio_.sum() * 100

    fig = px.scatter_3d(
        components, x=0, y=1, z=2, color=df['escrow'],
        title=f'Total Explained Variance: {total_var:.2f}%',
        labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
    )
    fig.show()


    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    print(test1, test2, sep='\n')
    print(test_pred1, test_pred2)

    acc = accuracy_score(y_test, y_pred)
    print(acc)

    # график
    #plot_data(X_train, y_train, 'clf')
    #plot_data(X_train, y_train, 'tree')


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
boxplot = df.boxplot()
plt.show()

X = df.iloc[:, [0, 1, 2, 3, 4, 5]].values
y = df.iloc[:, -1].values
