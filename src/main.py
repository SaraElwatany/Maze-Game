import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from model import train
from evaluation import evaluate_model
from preprocessing import preprocess, encode_labels






def main():

    """
    Main function to run the ML pipeline:
    - Sets up MLflow experiment
    - Loads and preprocesses data
    - Trains and evaluates multiple models with different hyperparameters
    - Logs performance metrics and confusion matrix for each run
    """

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Session 3 Experiment")

    # Load dataset and preprocess
    df = pd.read_csv("dataset/hand_landmarks_data.csv")
    X_train, X_test, y_train, y_test = preprocess(df)

    models = ['KNN', 'SVM', 'RandomForest', 'ExtremeGradientBoosting']

    model_params = {
        'KNN': [
            {'n_neighbors': 4, 'weights': 'distance', 'p': 2}
        ],
        'SVM': [
            {'kernel': 'linear', 'loss': 'hinge', 'c': 2, 'multi_class': 'ovr',
             'fit_intercept': True, 'intercept_scaling': 1, 'verbose': 1, 'random_state': 42, 'max_iter': 3000},
            {'kernel': 'rbf', 'gamma': 0.5, 'c': 370, 'random_state': 42, 'decision_function_shape': 'ovr'},
            {'kernel': 'poly', 'degree': 3, 'gamma': 10, 'c': 2, 'random_state': 42, 'decision_function_shape': 'ovr'}
        ],
        'RandomForest': [
            {'n_estimators': 500}
        ],
        'ExtremeGradientBoosting': [
            {'n_estimators': 500, 'learning_rate': 0.1, 'max_depth': 3}
        ]
    }

    run_id = 0
    for current_model in models:
        for current_params in model_params[current_model]:
            with mlflow.start_run(run_name=f'run_{run_id}', nested=True):
                try:
                    # Log hyperparameters
                    for key, val in current_params.items():
                        mlflow.log_param(key, val)

                    # Train and predict
                    model, _ = train(X_train, y_train, current_model, current_params)
                    y_pred = model.predict(X_test)

                    # Evaluate and log metrics
                    if current_model == 'ExtremeGradientBoosting':
                        y_test = encode_labels(y_test)

                    conf_mat, acc, precision, recall, f1 = evaluate_model(y_pred, y_test)
                    mlflow.log_metrics({
                                        "accuracy": acc,
                                        "precision": precision,
                                        "recall": recall,
                                        "f1_score": f1
                                    })

                    # Log metadata
                    mlflow.set_tag("version", "1.3")
                    mlflow.set_tag("model", current_model)

                    # Save and log confusion matrix plot
                    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
                    disp.plot()
                    artifact_path = 'plot_confusion_matrix.png'
                    plt.savefig(artifact_path)
                    mlflow.log_artifact(artifact_path, artifact_path="artifacts")
                    plt.close()


                    # Save & Log the model
                    mlflow.sklearn.log_model(model, artifact_path="Model")


                finally:
                    mlflow.end_run()

            run_id += 1


if __name__ == "__main__":
    main()

