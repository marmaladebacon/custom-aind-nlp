#for vectorizer
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
# joblib is an enhanced version of pickle that is more efficient for storing NumPy arrays

cache_dir = os.path.join("cache", "sentiment")  # where to store cache files
os.makedirs(cache_dir, exist_ok=True)  # ensure cache directory exists

def extract_BoW_features(words_train, words_test, vocabulary_size = 5000,
    cache_dir=cache_dir, cache_file="bow_features.pkl", force=False):

    cache_data = None
    # Try to open the cache file if string is given, and open in (r)ead and (b)inary mode
    if cache_file is not None and force==False:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = joblib.load(f)
            print("Read features from cache file:", cache_file)
        except:
            print("Unable to read from cache, creating new:", cache_file)
            pass

    if cache_data is None or force==True:
        #Fit Vectorizer to training docs and use it to tranform them
        #Remember to convert to array for a compact representation
        #Assume training docs have been preprocessed earlier and pass in dummy functions
        vectorizer = CountVectorizer(preprocessor=lambda x: x, tokenizer=lambda x:x, 
            max_features=vocabulary_size)
        features_train = vectorizer.fit_transform(words_train).toarray()
        
        #Apply the same vectorizer to transform the test documents(ignore unknown words)        
        features_test = vectorizer.transform(words_test).toarray()

        if cache_file is not None:
            vocabulary = vectorizer.vocabulary_
            cache_data = dict(features_train=features_train, features_test=features_test, 
                vocabulary=vocabulary)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                joblib.dump(cache_data, f)
                print("wrote features to cache file:", cache_file)
    else:
        #unpack data stored in cache file
        features_train, features_test, vocabulary = (cache_data["features_train"],
            cache_data["features_test"], cache_data["vocabulary"])

    return features_train, features_test, vocabulary


