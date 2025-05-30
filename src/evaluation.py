import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix







def evaluate_model(y_pred, y_test) -> tuple:
  
  """
  Function to evaluate the given predictions of a classification model based on several evaluation metrics like: accuracy, f1-score, precision, recall,
  and also visualizing the output predictions using the confusion matrix.

  Args:
    y_pred (numpy.ndarray): The predictions of the classification model to be evaluated.
    y_test (pd.DataFrame): The ground truth labels.

  Returns:
    tuple: (accuracy, precision, recall, f1) containing:
        - accuracy (float): The accuracy of the classification model. i.e Proportion of correctly classified samples.
        - precision (float): The precision of the classification model. i.e Proportion of correctly predicted positive samples out of all predicted positives.
        - recall (float): The recall () of the classification model. i.e Proportion of correctly predicted positive samples out of all actual positives.
        - f1 (float): The f1_score of the classification model. i.e Harmonic mean of precision and recall.
  """

  # Calculate accuracy of the overall performance
  accuracy = accuracy_score(y_test, y_pred)

  # Get weighted metrics
  precision = precision_score(y_test, y_pred, average='weighted')
  recall = recall_score(y_test, y_pred, average='weighted')
  f1 = f1_score(y_test, y_pred, average='weighted')
  cm = confusion_matrix(y_test, y_pred)

  # Print Metrics
  print(f"Accuracy: {accuracy * 100:.4f} %")
  print(f"Precision: {precision:.4f}")
  print(f"Recall: {recall:.4f}")
  print(f"F1-score: {f1:.4f}")


#   # Plot Confusion Matrix
#   plt.figure(figsize=(8, 6))
#   sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=set(y_test), yticklabels=set(y_test))
#   plt.xlabel('Predicted')
#   plt.ylabel('Actual')
#   plt.title('Confusion Matrix')
#   plt.show()


  return cm, accuracy, precision, recall, f1