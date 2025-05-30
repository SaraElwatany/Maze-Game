import mlflow
from sklearn import svm
from xgboost import XGBClassifier
from mlflow.data import from_pandas
from preprocessing import encode_labels
from mlflow.models import infer_signature
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier









def knn_model(X_train, y_train, n_neighbors=4, weights='distance', p=2):

    """
    Train and log a K-Nearest Neighbors (KNN) model using MLflow.

    Args:
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training labels.
        n_neighbors (int): Number of neighbors to use.
        weights (str): Weight function used in prediction ('uniform' or 'distance').
        p (int): Power parameter for the Minkowski metric (e.g., 2 for Euclidean).

    Returns:
        model (KNeighborsClassifier): Trained KNN model.
        params (dict): Dictionary of model parameters.
    """


    # Train a K-Nearest Neighbors Model, and having the following setup
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p, n_jobs=-1)
    model = model.fit(X_train, y_train)     # n_neighbors=4, weights=distance, p=3 with default par, Accuracy: 92.9114 %

    signature = infer_signature(X_train, y_train)
    mlflow.sklearn.log_model(model, 'Model', signature=signature, input_example=X_train)

    dataset = from_pandas(X_train, name="Hand Gestures Landmarks Data")
    mlflow.log_input(dataset, context="training")

    params = {'n_neighbors': n_neighbors, 'weights': weights, 'p': p}


    return model, params






def svm_model(X_train, y_train, kernel='rbf', loss='hinge', multi_class= 'ovr', decision_function_shape='ovr', c=2, gamma=0.5, degree=3, fit_intercept=True, intercept_scaling=1, verbose=1, random_state=42, max_iter=3000):

    """
    Train and log a Support Vector Machine (SVM) model using MLflow.

    Args:
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training labels.
        c (float): Regularization parameter.
        gamma (float): Kernel coefficient for 'rbf' and 'poly'.
        degree (int): Degree for the polynomial kernel.
        fit_intercept (bool): Whether to calculate the intercept for the model.
        intercept_scaling (float): Value for intercept scaling in linear SVMs.
        verbose (int): Enable verbose output.
        random_state (int): Random seed.
        max_iter (int): Maximum iterations.
        kernel (str): Kernel type ('linear', 'rbf', 'poly').

    Returns:
        model (SVC or LinearSVC): Trained SVM model.
        params (dict): Dictionary of model parameters.
    """


    if kernel == 'linear':
        model = svm.LinearSVC(loss=loss, C=c, multi_class=multi_class, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, verbose=verbose, random_state=random_state, max_iter=max_iter)
        model = model.fit(X_train, y_train)     # C=2, multi_class='ovr', fit_intercept=True, intercept_scaling=1, verbose=1, random_state=42, max_iter=3000, test_acc = 81.3632 %
        params = {'kernel': 'linear', 'loss': 'hinge','C':c, 'multi_class':'ovr', 'fit_intercept':fit_intercept, 'intercept_scaling':intercept_scaling, 'verbose':verbose, 'random_state':random_state, 'max_iter':max_iter}

    elif kernel == 'rbf':
        model = svm.SVC(kernel='rbf', gamma=gamma, C=c, decision_function_shape=decision_function_shape, random_state=random_state)
        model = model.fit(X_train, y_train)            # c= 370, gamma=0.5, test_acc = 97.2736 %
        params = {'kernel':'rbf', 'gamma':gamma, 'C':c,'random_state':random_state, 'decision_function_shape':'ovr'}

    elif kernel == 'poly':
        model = svm.SVC(kernel='poly', degree=degree, gamma=gamma, C=c, decision_function_shape=decision_function_shape, random_state=random_state)
        model = model.fit(X_train, y_train)           # degree=3, gamma=10, C=2,  Accuracy: 97.7994 %
        params = {'kernel':'poly', 'degree':degree, 'gamma':gamma, 'C':c,'random_state':random_state, 'decision_function_shape':'ovr'}


    signature = infer_signature(X_train, y_train)
    mlflow.sklearn.log_model(model, 'Model', signature=signature, input_example=X_train)

    dataset = from_pandas(X_train, name="Hand Gestures Landmarks Data")
    mlflow.log_input(dataset, context="training")


    return model, params






def random_forest_model(X_train, y_train, n_estimators=500):

    """
    Train and log a Random Forest model using MLflow.

    Args:
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training labels.
        n_estimators (int): Number of trees in the forest.

    Returns:
        model (RandomForestClassifier): Trained random forest model.
        params (dict): Dictionary of model parameters.
    """


    model = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    model = model.fit(X_train, y_train)       # n_estimators=500, default par, Accuracy: 95.0925 %

    signature = infer_signature(X_train, y_train)
    mlflow.sklearn.log_model(model, 'Model', signature=signature, input_example=X_train)

    dataset = from_pandas(X_train, name="Hand Gestures Landmarks Data")
    mlflow.log_input(dataset, context="training")

    params = {'n_estimators': n_estimators, 'random_state': 42}
    
    return model, params





def extreme_gradient_boosting_model(X_train, y_train, n_estimators=500, learning_rate=0.1, max_depth=3):

    """
    Train and log a Gradient Boosting (XGBoost) model using MLflow.

    Args:
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training labels.
        n_estimators (int): Number of boosting rounds (trees).
        learning_rate (float): Step size shrinkage.
        max_depth (int): Maximum tree depth.

    Returns:
        model (XGBClassifier): Trained XGBoost model.
        params (dict): Dictionary of model parameters.
    """

    # Train XGBoost Model, and having the following setup
    model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
    y_train_encoded = encode_labels(y_train)

    model = model.fit(X_train, y_train_encoded)         # n_estimators=500, learning_rate=0.1, max_depth=3, 97.6241 %

    signature = infer_signature(X_train, y_train)
    mlflow.sklearn.log_model(model, 'Model', signature=signature, input_example=X_train)

    dataset = from_pandas(X_train, name="Hand Gestures Landmarks Data")
    mlflow.log_input(dataset, context="training")

    params = {'n_estimators': n_estimators, 'learning_rate': learning_rate, 'max_depth': max_depth, 'random_state': 42}

    return model, params








def train(X_train, y_train, base_model, params):

    """
    Selects and trains one of the specified models based on input.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        base_model (str): Model type to train
        params (dict): Model parameters

    Returns:
        model: Trained model
        dict: Parameters used for training
    """

    if base_model == 'KNN':
        return knn_model(X_train, y_train, **params)

    elif base_model == 'SVM':
        return svm_model(X_train, y_train, **params)

    elif base_model == 'RandomForest':
        return random_forest_model(X_train, y_train, **params)
    
    elif base_model == 'ExtremeGradientBoosting':
        return extreme_gradient_boosting_model(X_train, y_train, **params)

    else:
        raise ValueError(f"Unknown model type: {base_model}")

