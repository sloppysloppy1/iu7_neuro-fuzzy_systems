import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

from mpl_toolkits.mplot3d import Axes3D

le, ms = LabelEncoder(), MinMaxScaler()

#categories = ['Vendor', 'Category', 'Item', 'Item Description', 'Price', 'Origin', 'Rating']
categories = ['product_title', 'ships_from_to', 'grams', 'quality', 'btc_price', 'cost_per_gram', 'cost_per_gram_pure', 'escrow', 'product_link']

def plt_compare(df):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,8))

    silhouette_scores = []
    elbow_scores = []
    bolduin_scores = []
    for i in range(2, 6):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 1000, n_init = 10, random_state = 0)
        kmeans.fit_predict(df)
        elbow_scores.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(df, kmeans.labels_))
        bolduin_scores.append(davies_bouldin_score(df, kmeans.labels_))

    plt.ylabel('WGCS')
    plt.xlabel('кол-во кластеров')
    ax1.plot(range(2,6), elbow_scores)
    ax2.plot(range(2,6), silhouette_scores)
    ax3.plot(range(2,6), bolduin_scores)
    ax1.set_title('метод локтя')
    ax2.set_title('метод силуэта')
    ax3.set_title('davis-bolduin')
    plt.show()

def get_eps(df):
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(df)
    distances, indices = nbrs.kneighbors(df)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances)
    plt.show()
    print(distances, indices)

def kmeans(df, clusters):
    fig = plt.figure()
    ax = Axes3D(fig)

    kmeans = KMeans(n_clusters = clusters, init = 'k-means++', max_iter = 1000, n_init = 10, random_state = 0)
    labels = kmeans.fit_predict(df)

    centroids = kmeans.cluster_centers_
    u_labels = np.unique(labels)
    for i in u_labels:
        ax.scatter(df.iloc[labels == i , 0], df.iloc[labels == i , 1], label = i)
        #axs[0].scatter(df.iloc[labels == i , 0], df.iloc[labels == i , 1], label = i)
        #axs[1,0].scatter(df.iloc[labels == i , 3], df.iloc[labels == i , 4], df.iloc[labels == i , 5], label = i)
    ax.scatter(centroids[:,0] , centroids[:,1] , s = 10, color = 'black')
    #axs[1,0].scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'black')
    plt.show()

def dbscan(df):
    fig = plt.figure()
    ax = Axes3D(fig)
    db = DBSCAN(eps=0.367, min_samples=18).fit(df) # 2 * число параметров
    labels = db.fit_predict(df)
    u_labels = np.unique(labels)
    for i in u_labels:
        ax.scatter(df.iloc[labels == i , 0], df.iloc[labels == i , 1], label = i)
        #axs[1].scatter(df.iloc[labels == i , 0], df.iloc[labels == i , 1], label = i)
        #axs[1,1].scatter(df.iloc[labels == i , 3], df.iloc[labels == i , 4], df.iloc[labels == i , 5], label = i)

    #axs[0].set_title('k средних')
    #axs[1].set_title('db scan')
    plt.show()

a = input()
#df = pd.read_csv('Agora.csv', delimiter=',', encoding ='latin-1')
df = pd.read_csv('dream_market_cocaine_listings.csv', delimiter=',', encoding ='latin-1')
df = df[categories]
#df.drop(['Destination', 'Remarks'], axis=1, inplace=True)

for category in categories:
    df[category] = le.fit_transform(df[category])

df = pd.DataFrame(ms.fit_transform(df), columns=categories)
print(df)

if a == 'k':
    plt_compare(df)
    kmeans(df, 4)
else:
    get_eps(df)
    dbscan(df)
