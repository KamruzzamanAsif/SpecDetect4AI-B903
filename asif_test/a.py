# imports
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error  
from sklearn.preprocessing import StandardScaler

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target vector.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.

    Returns:
    X_train, X_test, y_train, y_test: Split datasets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_data(X_train, X_test):    
    """
    Scale the feature data using StandardScaler.

    Parameters:
    X_train (array-like): Training feature matrix.
    X_test (array-like): Testing feature matrix.

    Returns:
    X_train_scaled, X_test_scaled: Scaled training and testing feature matrices.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_linear_regression(X, y, test_size=0.2, random_state=42):
    """
    Train a simple linear regression model.

    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target vector.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.

    Returns:
    model: Trained Linear Regression model.
    mse: Mean Squared Error on the test set.
    """
    # fit transform? scale data if needed
    # For linear regression, scaling is not strictly necessary, but can be done if desired
    #move these to another function
    X_train, X_test = scale_data(X, X)  # Note: This is just a placeholder; scaling should be done after splitting

    # Split the dataset into training and testing sets
    # X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Create a Linear Regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)

    return model, mse


# def scale_data(X_train, X_test):    
#     """
#     Scale the feature data using StandardScaler.

#     Parameters:
#     X_train (array-like): Training feature matrix.
#     X_test (array-like): Testing feature matrix.

#     Returns:
#     X_train_scaled, X_test_scaled: Scaled training and testing feature matrices.
#     """
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#     return X_train_scaled, X_test_scaled

if __name__ == "__main__":
    # Example usage with dummy data
    X = np.random.rand(100, 5)
    y = np.random.rand(100)

    model, mse = train_linear_regression(X, y)
    print(f"Trained Linear Regression Model: {model}")
    print(f"Mean Squared Error on test set: {mse}")