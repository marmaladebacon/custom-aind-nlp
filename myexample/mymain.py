import matplotlib.pyplot as plt
from libs.cachebuilder import preprocess_data
from libs.extractbowfeatures import extract_BoW_features
from libs.normalizefeatures import normalize
import os
import glob
import re
import nltk
import numpy as np
nltk.download("stopwords") #download list of stopwords
from nltk.corpus import stopwords #import stopwords
from nltk.stem.porter import *
stemmer = PorterStemmer()

from wordcloud import WordCloud, STOPWORDS
from bs4 import BeautifulSoup
from sklearn.utils import shuffle

def main():
    print('my main')
    data = read_imdb_data()
    print("IMDb reviews: train = {} pos / {} neg, test = {} pos / {} neg".format(
        len(data['train']['pos']), len(data['train']['neg']),
        len(data['test']['pos']), len(data['test']['neg'])))

    create_word_cloud(data,'pos')
    create_word_cloud(data,'neg')

    data_train, data_test, labels_train, labels_test = prepare_imdb_data(data)
    
    words_train, words_test, labels_train, labels_test = preprocess_data(data_train, data_test, labels_train, labels_test, clean_words)

    features_train, features_test, vocabulary = extract_BoW_features(words_train, words_test,force=True)
    print(features_train[:2])
    #vocabulary is a dict mapping the word to the index in the feature array
    #From the docs "A mapping of terms to feature indices."
    #print(vocabulary)

    #plt.plot(features_train[5,:])
    #plt.xlabel('Word')
    #plt.ylabel('Count')
    #plt.show()

    #Sparsity (0 to 1)    
    total_features=features_train.shape[1]*features_train.shape[0]
    print("Sparsity:", (total_features- np.count_nonzero(features_train))/total_features)

    # Find number of occurrences for each word in the training set
    word_freq = features_train.sum(axis=0)

    # Sort it in descending order
    sorted_word_freq = np.sort(word_freq)[::-1]

    # Plot 
    #plt.plot(sorted_word_freq)
    #plt.gca().set_xscale('log')
    #plt.gca().set_yscale('log')
    #plt.xlabel('Rank')
    #plt.ylabel('Number of occurrences')
    #plt.show()

    #Array of word frequencies
    #print(word_freq)
    #most frequent word
    print(sorted_word_freq[0])
    #secondmost frequent word
    print(sorted_word_freq[1])

    print(features_train[0])
    print(len(features_train))
    print(len(features_train[0]))
    features_train, features_test = normalize(features_train, features_test)
    print(features_train[0])

    

def read_imdb_data(data_dir='minidata/imdb-reviews'):
    """Read IMDb movie reviews from given directory.

    Directory structure expected:
    - minidata/
        -imdb-reviews/
            - train/
                - pos/
                - neg/
            - test/
                - pos/
                - neg/

    """
    data = {}
    labels = {}

    for data_type in ['train', 'test']:
        data[data_type] = {}

        for sentiment in ['pos', 'neg']:
            data[data_type][sentiment] = []

            #Fetch all necessary raw files
            path = os.path.join(data_dir, data_type, sentiment, '*.txt')
            files = glob.glob(path)

            for f in files:
                with open(f, encoding='utf-8') as review:
                    data[data_type][sentiment].append(review.read())



    return data

def create_word_cloud(data, sentiment):
    #combine all reviews
    combined_text = " ".join([review for review in data['train'][sentiment]])
    #produce word cloud minus common stop words
    wc = WordCloud(background_color='white', max_words=50,
                    stopwords=STOPWORDS.update(['br', 'film','movie']))
    generated_wc = wc.generate(combined_text)
    #plt.imshow(generated_wc)
    #plt.axis('off')
    #plt.show()
    generated_wc.to_file("./img/test-{}.png".format(sentiment))

def prepare_imdb_data(data):
    """Prepare training and test sets from IMDb movie reviews."""
    data_labels = {}
    data_labels['train'] = []
    data_labels['test'] = []

    # TODO: Combine positive and negative reviews and labels
    for dt in ['train', 'test']:
        for sentiment in ['pos', 'neg']:
            for element in data[dt][sentiment]:
                data_labels[dt].append((element, sentiment))

    # TODO: Shuffle reviews and corresponding labels within training and test sets
    for dt in ['train', 'test']:
        data_labels[dt] = shuffle(data_labels[dt])

    data_train = [e[0] for e in data_labels['train']]
    labels_train = [e[1] for e in data_labels['train']]
    data_test = [e[0] for e in data_labels['test']]
    labels_test = [e[1] for e in data_labels['test']]

    # Return a unified training data, test data, training labels, test labets
    return data_train, data_test, labels_train, labels_test

def clean_words(single_raw):
    # TODO: Remove HTML tags and non-letters,
    #       convert to lowercase, tokenize,
    #       remove stopwords and stem

    #Remove html tags
    soup = BeautifulSoup(single_raw, "html5lib")
    words = soup.get_text()
    #Remove non-letters
    words = re.sub(r"[^a-zA-Z0-9]", " ", words)
    #lowercase
    words = words.lower()
    #Tokenize
    words = words.split()
    #Remove stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    #Stem
    words = [stemmer.stem(w) for w in words]

    # Return final list of words
    return words

if __name__ == "__main__":
    main()