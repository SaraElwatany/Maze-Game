def center_landmarks(col, landmarks_df):
    col_name = col.name
    if 'x' in col_name:
        col = col - landmarks_df['x1']
    elif 'y' in col_name:
        col = col - landmarks_df['y1']
    return col



def normalize_landmarks(col, landmarks_df):
    col_name = col.name
    if 'x' in col_name:
        col = col / landmarks_df['x13']
    elif 'y' in col_name:
        col = col / landmarks_df['y13']
    return col