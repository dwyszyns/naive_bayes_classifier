import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from typing import Callable


def uniform_probability(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.where(np.logical_and(a[0] <= x, x <= b[0]), 1 / (b[0] - a[0]), 0)


def gauss_probability(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    exponent = np.exp(-((x - mean) ** 2) / (2 * std**2))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent


class NaiveBayesClassifier:
    def __init__(self, distribution_func: Callable):
        self.distribution_func = distribution_func
        self.means = []
        self.stds = []
        self.priors = []

    def compute_class_parameters(self, X: np.ndarray, y: np.ndarray) -> None:
        classes = np.unique(y)

        for c in classes:
            X_class = X[y == c]
            mean = np.mean(X_class, axis=0)
            std = np.std(X_class, axis=0)
            prior = len(X_class) / len(X)
            self.means.append(mean)
            self.stds.append(std)
            self.priors.append(prior)

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_predicted = []

        for i in range(len(X)):
            posteriors = []

            for j in range(len(self.means)):
                likelihoods = self.distribution_func(X[i], self.means[j], self.stds[j])
                posterior = np.prod(likelihoods) * self.priors[j]
                posteriors.append(posterior)

            predicted_class = np.argmax(posteriors)
            y_predicted.append(predicted_class)

        return np.array(y_predicted)


def run_classifier(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, distribution: Callable) -> float:
    classifier = NaiveBayesClassifier(distribution)
    classifier.compute_class_parameters(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    distributions = [uniform_probability, gauss_probability]

    for distribution in distributions:
        accuracy = run_classifier(X_train, y_train, X_test, y_test, distribution)
        print(f"Naive Bayes classifier - {distribution.__name__}: accuracy={accuracy}\n")
