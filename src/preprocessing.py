import mlflow
import pandas as pd
from pickle import dump
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split





# We need to center by subtracting the wrist position, followed by averaging with the middle finger tip position
# Wrist = 0, index (x1, y1, z1)
# Middle finger tip = 12, index (x13, y13, z13)

def center_landmarks(col, landmarks_df):

    """
    Function to center the coordinates by subtracting from the wrist coordinates. Ignore z coordinates as it is already scaled.

    Args:
      col (pandas.series) : The dataframe column we want to center, as a pandas series. Center by subtracting wrist position.
      landmarks_df (pd.DataFrame): DataFrame containing all landmark points.

    Returns:
      col (pandas.series) : The centered dataframe column, as a pandas series.
    """

    col_name = col.name       # Extract column name

    # Check if it is the x or y coordinate first, otherwise neglect the column
    if 'x' in col_name:
      col = (col - landmarks_df['x1'])

    elif 'y' in col_name:
      col = (col - landmarks_df['y1'])

    return col






def normalize_landmarks(col, landmarks_df):

    """
    Function to normalize the centered coordinates by dividing with the mid-finger tip coordinates. Ignore z coordinates as it is already scaled.

    Args:
      col (pandas.series) : The dataframe column we want to normalize, as a pandas series. Normalize by dividing middle finger tip position.
      landmarks_df (pd.DataFrame): DataFrame containing all landmark points.

    Returns:
      col (pandas.series) : The normalized dataframe column, as a pandas series.
    """

    col_name = col.name       # Extract column name

    # Check if it is the x or y coordinate first, otherwise neglect the column
    if 'x' in col_name:
      col = col / landmarks_df['x13']

    elif 'y' in col_name:
      col = col / landmarks_df['y13']

    return col






def encode_labels(y_train):

    """
    Encode class labels into numeric format using LabelEncoder and log the encoder as an MLflow artifact.

    Args:
        y_train (pd.Series): Series containing target class labels.

    Returns:
        np.ndarray: Encoded class labels as integers.
    """

    # Encode labels, since xgboost works with numbers
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)  # Convert labels to numbers

    # Save encodings for testing
    artifact_path = 'xgb_label_encoder.pkl'
    dump(label_encoder, open(artifact_path, 'wb'))         # Save the scaler

    # Log the transformer as an artifact
    mlflow.log_artifact(artifact_path, artifact_path="artifacts")

    return y_train_encoded







def preprocess(landmarks_df):

    """
    Preprocess the landmark dataset by centering and normalizing x and y coordinates,
    then split into training and testing sets.

    Args:
        landmarks_df (pd.DataFrame): DataFrame containing landmark coordinates and a 'label' column.

    Returns:
        X_train (pd.DataFrame): Training features after preprocessing.
        X_test (pd.DataFrame): Testing features after preprocessing.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.
    """


    # Copy the dataframe, to avoid information loss
    landmarks_df_copy = landmarks_df.copy()

    # Center the coordinates
    centered_landmarks_df_copy = landmarks_df_copy.apply(lambda col: center_landmarks(col, landmarks_df_copy), axis=0)

    # Apply normalization to the centered coordinates
    normalized_landmarks_df = centered_landmarks_df_copy.apply(lambda col: normalize_landmarks(col, landmarks_df_copy), axis=0)


    # Split the Dataset to training & testing sets, with 80% of the dataset for training & the rest (20%) for testing
    X_train, X_test, y_train, y_test = train_test_split(normalized_landmarks_df.drop(columns=['label'], inplace=False), normalized_landmarks_df['label'], test_size=0.2, random_state=42)


    return X_train, X_test, y_train, y_test