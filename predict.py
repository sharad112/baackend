import joblib
import numpy as np


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)

        if len(unique_classes) == 1 or (
            self.max_depth is not None and depth == self.max_depth
        ):
            return {"class": unique_classes[0], "count": len(y)}

        if num_features == 0:
            class_counts = [len(y[y == cls]) for cls in unique_classes]
            return {"class": unique_classes[np.argmax(class_counts)], "count": len(y)}

        feature_index, threshold = self._find_best_split(X, y)

        if feature_index is None:
            class_counts = [len(y[y == cls]) for cls in unique_classes]
            return {"class": unique_classes[np.argmax(class_counts)], "count": len(y)}

        indices_left = X[:, feature_index] <= threshold
        X_left, y_left = X[indices_left], y[indices_left]
        X_right, y_right = X[~indices_left], y[~indices_left]

        left_tree = self._grow_tree(X_left, y_left, depth + 1)
        right_tree = self._grow_tree(X_right, y_right, depth + 1)

        return {
            "feature_index": feature_index,
            "threshold": threshold,
            "left": left_tree,
            "right": right_tree,
        }

    def _find_best_split(self, X, y):
        num_samples, num_features = X.shape
        if num_samples <= 1:
            return None, None

        gini = np.inf
        for feature_index in range(num_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                indices_left = X[:, feature_index] <= threshold
                indices_right = ~indices_left
                gini_left = self._calculate_gini(y[indices_left])
                gini_right = self._calculate_gini(y[indices_right])
                weighted_gini = (len(y[indices_left]) / num_samples) * gini_left + (
                    len(y[indices_right]) / num_samples
                ) * gini_right

                if weighted_gini < gini:
                    gini = weighted_gini
                    best_feature, best_threshold = feature_index, threshold

        return best_feature, best_threshold

    def _calculate_gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities**2)
        return gini

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, node):
        if "class" in node:
            return node["class"]
        if x[node["feature_index"]] <= node["threshold"]:
            return self._predict_tree(x, node["left"])
        else:
            return self._predict_tree(x, node["right"])


class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.estimators = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            indices = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap, y_bootstrap = X[indices], y[indices]

            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_bootstrap, y_bootstrap)
            self.estimators.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.estimators])
        return np.array(
            [
                np.bincount(predictions[:, i]).argmax()
                for i in range(predictions.shape[1])
            ]
        )


def load_models():
    # Load the Random Forest model
    rf_model = joblib.load("random_forest_model.joblib")

    # Load the Decision Tree model
    dt_model = DecisionTree(max_depth=10)  # Assuming DecisionTree class is available
    dt_model.tree = joblib.load("decision_tree.joblib")

    return rf_model, dt_model


def crop_prediction_rf(user_data):
    # Load the Random Forest model
    rf_model, _ = load_models()

    # Predict crop using Random Forest model
    crop_label = rf_model.predict(np.array(user_data).reshape(1, -1))[0]

    # Map crop label to crop name
    label_to_crop = {
        0: "muskmelon",
        1: "watermelon",
        2: "papaya",
        3: "apple",
        4: "mango",
        5: "mothbeans",
        6: "mugbean",
        7: "lentil",
        8: "blackgram",
        9: "coconut",
        10: "pomegranate",
        11: "jute",
        12: "maize",
        13: "coffee",
        14: "orange",
        15: "chickpea",
        16: "peigonpeas",
        17: "rice",
        18: "kidneybeans",
        19: "grapes",
        20: "cotton",
        21: "banana",
    }

    predicted_crop = label_to_crop.get(crop_label)
    return predicted_crop


def get_user_input():
    print("Enter details for the new data point:")
    N = int(input("Enter the value of N: "))
    P = int(input("Enter the value of P: "))
    K = int(input("Enter the value of K: "))
    temperature = float(input("Enter the value of temperature (in Celsius): "))
    humidity = float(input("Enter the value of humidity: "))
    ph = float(input("Enter the pH value: "))
    rainfall = float(input("Enter the amount of rainfall: "))

    user_data = np.array([N, P, K, temperature, humidity, ph, rainfall])

    return user_data


def crop_prediction_dt(user_data):
    # Load the Decision Tree model
    _, dt_model = load_models()

    # Predict crop using Decision Tree model
    crop_label = dt_model.predict(np.array(user_data).reshape(1, -1))[0]

    # Map crop label to crop name
    label_to_crop = {
        0: "muskmelon",
        1: "watermelon",
        2: "papaya",
        3: "apple",
        4: "mango",
        5: "mothbeans",
        6: "mugbean",
        7: "lentil",
        8: "blackgram",
        9: "coconut",
        10: "pomegranate",
        11: "jute",
        12: "maize",
        13: "coffee",
        14: "orange",
        15: "chickpea",
        16: "peigonpeas",
        17: "rice",
        18: "kidneybeans",
        19: "grapes",
        20: "cotton",
        21: "banana",
    }

    predicted_crop = label_to_crop.get(crop_label)
    return predicted_crop
