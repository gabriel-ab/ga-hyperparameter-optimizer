import pandas as pd
import numpy as np

from aes.cleaners import TextCleaner
from aes.metrics import mean_squared_error
from aes.utils import random_split_df
from aes.vectorizers import TfidfVectorizer


class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.tree_ = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if depth == self.max_depth or n_samples < self.min_samples_split:
            return LeafNode(y)

        feature_idxs = np.random.choice(n_features, self.n_features_, replace=False)
        best_feature, best_threshold = self._best_split(X, y, feature_idxs)

        left_idxs = X[:, best_feature] <= best_threshold
        left_tree = self._build_tree(X[left_idxs], y[left_idxs], depth+1)

        right_idxs = X[:, best_feature] > best_threshold
        right_tree = self._build_tree(X[right_idxs], y[right_idxs], depth+1)

        return DecisionNode(best_feature, best_threshold, left_tree, right_tree)

    def _best_split(self, X, y, feature_idxs):
        best_gain = -np.inf
        split_idx, split_threshold = None, None
        for feature_idx in feature_idxs:
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature_idx], threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def _information_gain(self, y, feature, threshold):
        parent_entropy = self._entropy(y)
        left_idxs = feature <= threshold
        right_idxs = feature > threshold
        if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
            return 0
        left_entropy = self._entropy(y[left_idxs])
        right_entropy = self._entropy(y[right_idxs])
        children_entropy = (np.sum(left_idxs)/len(y))*left_entropy + (np.sum(right_idxs)/len(y))*right_entropy
        return parent_entropy - children_entropy

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def _predict(self, inputs):
        node = self.tree_
        while isinstance(node, DecisionNode):
            if inputs[node.feature_] <= node.threshold_:
                node = node.left_
            else:
                node = node.right_
        return node.value_

class DecisionNode:
    def __init__(self, feature, threshold, left, right):
        self.feature_ = feature
        self.threshold_ = threshold
        self.left_ = left
        self.right_ = right

class LeafNode:
    def __init__(self, value):
        self.value_ = np.mean(value)


def run(max_depth: int, min_samples_split: int, max_len: int):
    dataset = pd.read_parquet("data/dataset.parquet")
    train_set, test_set = random_split_df(dataset, 0.2)

    y_train = train_set["score"].to_list()
    y_test = test_set["score"].to_list()

    vectorizer = TfidfVectorizer(max_len=max_len)
    cleaner = TextCleaner()

    train_documents = [cleaner(row["text"]) for _, row in train_set.iterrows()]
    test_documents = [cleaner(row["text"]) for _, row in test_set.iterrows()]

    vectorizer.fit([cleaner(row["text"]) for _, row in dataset.iterrows()])

    train_encoded_documents = vectorizer.encode(train_documents)
    test_encoded_documents = vectorizer.encode(test_documents)

    tree = DecisionTreeRegressor(max_depth, min_samples_split)
    tree.fit(np.array(train_encoded_documents), y_train)
    y_pred = tree.predict(test_encoded_documents)

    mse = mean_squared_error(y_test, y_pred)
    
    return mse
