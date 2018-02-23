import sklearn.preprocessing as pr

def normalize(features_train, features_test):
    #by default normalization is the l2 norm 'sum of the squares of the elements gives one'
    features_train = pr.normalize(features_train)
    features_test = pr.normalize(features_test)
    return features_train, features_test

