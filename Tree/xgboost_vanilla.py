import numpy as np


class XGBoostScratch:
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        reg_lambda=1.0,
    ):
        # number of trees or number of boosting rounds
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        # regularization parameter (L2 regularization term on weights)
        self.reg_lambda = reg_lambda
        self.trees = []
        self.gamma = 0  # regularization parameter (complexity penalty)

    # define the class of decision tree - build a tree every iteration
    class DecisionTree:
        def __init__(self, max_depth, min_samples_split, reg_lambda):
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.reg_lambda = reg_lambda
            self.tree = None

        def fit(self, X, y, residuals):
            """
            fit the decision tree to the data with the given residuals
            """
            self.tree = self._build_tree(X, y, residuals, depth=0)
            return self

        def _build_tree(self, X, y, residuals, depth):
            """
            build the current decision tree with given residuals
            by recursively finding the best split and building subtrees vertically
            """
            # if the tree if not at max depth
            # and the number of samples is greater than the minimum samples for a split
            if depth < self.max_depth and len(y) >= self.min_samples_split:
                best_split = self._find_best_split(X, residuals)
                if best_split:
                    left_indices = best_split["left_indices"]
                    right_indices = best_split["right_indices"]

                    # keep building tree for each splited left and right tree until
                    # still be able to find best split
                    # and current depth < max depth
                    # and still enough sample at the child node
                    left_tree = self._build_tree(
                        X[left_indices],
                        y[left_indices],
                        residuals[left_indices],
                        depth + 1,
                    )
                    right_tree = self._build_tree(
                        X[right_indices],
                        y[right_indices],
                        residuals[right_indices],
                        depth + 1,
                    )
                    return {
                        "feature_index": best_split["feature_index"],
                        "threshold": best_split["threshold"],
                        "left": left_tree,
                        "right": right_tree,
                    }
            # compute the final loss at all final leaf
            leaf_value = self._compute_leaf_value(residuals)
            return {"leaf_value": leaf_value}

        def _find_best_split(self, X, residuals):
            """
            find the best split for the current node
            by iterating over all features and its candidate thresholds

            return a dictionary of best split among all features.
            just return a split of "the best one" feature
            """
            best_split = None
            best_gain = float("-inf")
            # m samples and n features
            m, n = X.shape

            # iterate over all features
            for feature_index in range(n):

                # find unique thresholds for the feature - exact greedy algorithm
                # loop through all the unique values in the feature
                thresholds = np.unique(X[:, feature_index])
                for threshold in thresholds:

                    # split the data into left and right based on the threshold
                    left_indices = np.where(X[:, feature_index] <= threshold)[0]
                    right_indices = np.where(X[:, feature_index] > threshold)[0]

                    if len(left_indices) > 0 and len(right_indices) > 0:
                        gain = self._compute_gain(
                            residuals, left_indices, right_indices
                        )
                        if gain > best_gain:
                            best_gain = gain
                            best_split = {
                                "feature_index": feature_index,
                                "threshold": threshold,
                                "left_indices": left_indices,
                                "right_indices": right_indices,
                            }
            return best_split

        def _compute_gain(self, residuals, left_indices, right_indices):
            """
            compute the gain in loss after the split
            """

            # split the residuals into left and right based on the indices
            # that generated by current split (left_indices, right_indices)
            left_residuals = residuals[left_indices]
            right_residuals = residuals[right_indices]

            # try to find the convex of the loss function
            # current gain = loss(parent) - (loss(left) + loss(right))

            # theoretically, loss can be estimated with G^2 / (H + lambda)
            # TODO - check the 2nd order derivative
            # where G is the sum of gradients (1st order derivative)
            # and H is the sum of hessians (2nd order derivative)

            # while the loss function is the squared error loss = (y - y_pred)^2
            # the gradient is the residual and the hessian is 1
            # Therefore, loss := G^2 / (H + lambda) = sigma[residual] ^ 2 / (n + lambda)
            # TODO - WHERE IS THE 0.5 COME FROM??
            gain = 0.5 * (
                (np.sum(left_residuals) ** 2 / (len(left_residuals) + self.reg_lambda))
                + (
                    np.sum(right_residuals) ** 2
                    / (len(right_residuals) + self.reg_lambda)
                )
            ) - (np.sum(residuals) ** 2 / (len(residuals) + self.reg_lambda))
            return gain

        def _compute_leaf_value(self, residuals):
            return np.sum(residuals) / (len(residuals) + self.reg_lambda)

        def predict(self, X):
            return np.array([self._predict_one(x, self.tree) for x in X])

        def _predict_one(self, x, tree):
            if "leaf_value" in tree:
                return tree["leaf_value"]
            feature_index = tree["feature_index"]
            threshold = tree["threshold"]
            if x[feature_index] <= threshold:
                return self._predict_one(x, tree["left"])
            else:
                return self._predict_one(x, tree["right"])

    def fit(self, X, y):

        # start with all 0 for all predictions
        y_pred = np.zeros(len(y))
        # build n trees based on the n_estimators value given
        for _ in range(self.n_estimators):
            # calculate the residual from last tree prediction
            residuals = y - y_pred
            # fit each tree
            tree = self.DecisionTree(
                self.max_depth, self.min_samples_split, self.reg_lambda
            )
            tree.fit(X, y, residuals)
            # since learnt on residual - the new y prediction is the aggregation of new tree * learning rate
            y_pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred


# Usage Example
if __name__ == "__main__":
    from sklearn.datasets import load_boston
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    # Load dataset
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and train the model
    model = XGBoostScratch(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=10,
        reg_lambda=1.0,
    )
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")


# This code was generated with the assistance of ChatGPT, an AI language model created by OpenAI.
# Model used: GPT-4 (ChatGPT)
# Date: [insert date you generated the code]
# For more information, visit https://www.openai.com/
