import pandas as pd
import numpy as np
import re, string
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import Perceptron

import gensim
from gensim.models import Word2Vec

from prettytable import PrettyTable

wl = WordNetLemmatizer()
v = TfidfVectorizer()
le = LabelEncoder()

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
colors = {0: 'red', 1: 'green'}
resolution = 6e-3
tab = PrettyTable(['type', 'conf_matrix', '0', '1', 'accuracy', 'R', 'P', 'F_measure'])

def preprocess(text):
    text = text.lower().strip()
    text = re.compile('<.*?>').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]',' ',text)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text)
    text = re.sub(r'\s+',' ',text)
    return text

def stopword(string):
    return ' '.join([i for i in string.split() if i not in stopwords.words('english')])

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string))
    return ' '.join([wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)])

def plot(X_test, y_test, X_train, y_train, ax):
    clf = Perceptron(random_state=1, eta0 = 10, max_iter = 10000, alpha = 0.001)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

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
        ax.scatter(X_test[indices, 0], X_test[indices, 1], c=colors[label], alpha=0.8, cmap=cmap_light,
                    label='class {}'.format(label))

    _confusion_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = _confusion_matrix.ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    R = tp / (tp + fn)
    P = tp / (tp + fp)
    F_measure = tp / (2 * tp + fp + fn)
    tab.add_row(['', 0] + list(_confusion_matrix[0]) + [''] * 4)
    tab.add_row(['perceptron', 1] + list(_confusion_matrix[1]) + [acc, R, P, F_measure])

    return F_measure


df = pd.read_csv('Womens Clothing E-Commerce Reviews.csv', delimiter=',', encoding ='latin-1')
df = df[['Review Text', 'Recommended IND']].sort_values(by=['Recommended IND'])
df = df.iloc[3500:5000,].dropna()
df['clean_text'] = df['Review Text'].apply(lambda x: (stopword(lemmatizer(preprocess(x)))))
df = df[['Review Text', 'clean_text', 'Recommended IND']]
print(df)
X = df.iloc[:,1].values
y = df.iloc[:,2].values

X = v.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=1)

pca = PCA(n_components=2, random_state=1)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
ax.set_title('perceptron')
plot(X_test, y_test, X_train, y_train, ax)
print(tab)

plt.show()
