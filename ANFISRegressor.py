"""
Created on Tue May  6 21:22:36 2024

@author: Mostafa Haghighi
Email : Mostafahaghighi.ce@gmail.com

Adaptive Neuro-Fuzzy Inference System regression
ANFISRegressor
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
# optimal values from Gridsearch (2, 0.2, 400)

class ANFISRegressor:
    def __init__(self, n_membership=2, learning_rate=0.2, epochs=400):
        self.n_membership = n_membership
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.membership_params = None
        self.consequent_params = None

    def gaussian_mf(self, x, mean, sigma):
        return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

    def initialize_parameters(self, X):
        n_features = X.shape[1]

        # Initialize membership function parameters
        ranges = [(X[:, i].min(), X[:, i].max()) for i in range(n_features)]
        means = []
        sigmas = []

        for feature_range in ranges:
            feature_means = np.linspace(feature_range[0], feature_range[1], self.n_membership)
            feature_sigma = (feature_range[1] - feature_range[0]) / (2 * self.n_membership)
            means.append(feature_means)
            sigmas.append(np.ones(self.n_membership) * feature_sigma)

        self.membership_params = {'means': means, 'sigmas': sigmas}

        # Initialize consequent parameters
        n_rules = self.n_membership ** n_features
        self.consequent_params = np.random.randn(n_rules, n_features + 1) * 0.1

    def compute_firing_strengths(self, X):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_rules = self.n_membership ** n_features

        # Compute membership degrees
        memberships = []
        for i in range(n_features):
            feature_memberships = np.zeros((n_samples, self.n_membership))
            for j in range(self.n_membership):
                feature_memberships[:, j] = self.gaussian_mf(
                    X[:, i],
                    self.membership_params['means'][i][j],
                    self.membership_params['sigmas'][i][j]
                )
            memberships.append(feature_memberships)

        # Compute firing strengths using product t-norm
        firing_strengths = np.ones((n_samples, n_rules))
        for i in range(n_rules):
            rule_memberships = []
            temp = i
            for j in range(n_features):
                mf_idx = temp % self.n_membership
                temp = temp // self.n_membership
                rule_memberships.append(memberships[j][:, mf_idx])
            firing_strengths[:, i] = np.prod(rule_memberships, axis=0)

        # Normalize firing strengths
        sum_firing = np.sum(firing_strengths, axis=1, keepdims=True)
        normalized_firing = firing_strengths / (sum_firing + 1e-10)

        return normalized_firing

    def forward_pass(self, X):
        n_samples = X.shape[0]
        X_aug = np.hstack([X, np.ones((n_samples, 1))])  # Add bias term

        # Layer 1-3: Compute normalized firing strengths
        normalized_firing = self.compute_firing_strengths(X)

        # Layer 4-5: Compute consequent outputs and final prediction
        consequent_outputs = np.dot(X_aug, self.consequent_params.T)
        y_pred = np.sum(normalized_firing * consequent_outputs, axis=1)

        return y_pred, normalized_firing, X_aug

    def fit(self, X, y):
        self.initialize_parameters(X)

        for epoch in range(self.epochs):
            # Forward pass
            y_pred, normalized_firing, X_aug = self.forward_pass(X)

            # Backward pass
            error = y - y_pred

            # Update consequent parameters
            for i in range(len(self.consequent_params)):
                # Reshape error and normalized_firing for proper broadcasting
                error_reshaped = error.reshape(-1, 1)
                firing_reshaped = normalized_firing[:, i].reshape(-1, 1)

                # Calculate gradient with proper broadcasting
                gradient = -2 * np.mean(error_reshaped * firing_reshaped * X_aug, axis=0)
                self.consequent_params[i] -= self.learning_rate * gradient

            if epoch % 10 == 0:
                mse = np.mean(error ** 2)
                print(f"Epoch {epoch}, MSE: {mse:.6f}")

    def predict(self, X):
        y_pred, _, _ = self.forward_pass(X)
        return y_pred

# Example usage
def main():
    # Load data


    df = pd.read_csv('data.csv')

    train_dataset = df.sample(frac=0.8, random_state=42)
    test_dataset = df.drop(train_dataset.index)
    test_dataset.head()
    xtrain = train_dataset.copy()
    xtest = test_dataset.copy()

    xtrain = df.drop(['Features'],axis=1)
    xtest  =  df.drop(['Features'],axis=1)

    ytrain = xtrain.pop('Target')
    ytest = xtest.pop('Target')

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Create and train ANFIS model
    anfis = ANFISRegressor(n_membership=2, learning_rate=0.2, epochs=400)
    anfis.fit(X_train, y_train)

    # Make predictions

    y_pred_test = anfis.predict(X_test)
    mse = np.mean((y_test - y_pred_test) ** 2)
    print(f"Test MSE: {mse:.6f}")
    r2 = r2_score(y_test, y_pred_test )
    print("R-squared for test: {:.4f}".format(r2))


    y_pred_train = anfis.predict(X_train)
    r2_train = r2_score(y_train, y_pred_train )
    print("R-squared for train : {:.4f}".format(r2_train))


    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.7)
    plt.xlabel('Experimental Values')
    plt.ylabel('Predicted Values')
    plt.title('ANFIS Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-')  # line y=x
    plt.show()


if __name__ == "__main__":
    main()
