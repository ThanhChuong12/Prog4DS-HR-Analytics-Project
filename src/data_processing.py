import numpy as np
import os

# HELPER FUNCTIONS
# Remove unwanted characters (e.g. commas)
def clean_currency_string(arr):
    return np.char.replace(arr.astype(str), ',', '')

# Fill NaN values
def fill_missing_numerical(arr, strategy = 'median'):
    arr = arr.astype(float)
    mask_nan = np.isnan(arr)
    if strategy == 'median':
        fill_val = np.nanmedian(arr)
    elif strategy == 'mean':
        fill_val = np.nanmean(arr)
    else:
        fill_val = 0.0
    arr[mask_nan] = fill_val
    return arr

# Map categorical values to numbers
def map_ordinal(arr, mapping):
    arr = arr.astype(str)
    arr = np.char.strip(arr)
    conditions = [(arr == key) for key in mapping.keys()]
    choices = list(mapping.values())
    return np.select(conditions, choices, default = np.nan)


def one_hot(arr):
    # Find unique values and their inverse indices
    uniques, inverse_indices = np.unique(arr, return_inverse=True)
    # Create identity matrix and select rows based on inverse indices
    one_hot_matrix = np.eye(len(uniques))[inverse_indices]
    
    return one_hot_matrix, uniques


# MAIN PREPROCESSING
def process_hr_data(data_raw):
    header = data_raw[0]
    body = data_raw[1:]
    col_idx = {name: i for i, name in enumerate(header)}
    processed = {}

    # NUMERICAL FEATURES
    # 1. City Development Index
    cdi = body[:, col_idx['city_development_index']]
    cdi = np.where(cdi == '', 'nan', cdi)
    processed['city_development_index'] = cdi.astype(float)

    # 2. Training Hours
    th = body[:, col_idx['training_hours']]
    th = np.where(th == '', 'nan', th)
    processed['training_hours'] = th.astype(float)

    # ORDINAL FEATURES
    # 3. Experience
    exp_map = {'<1': 0, '>20': 21}
    exp_map.update({str(i): i for i in range(1, 21)})
    processed['experience'] = map_ordinal(body[:, col_idx['experience']], exp_map)

    # 4. Company Size (Fixing '10/49' format issue)
    size_raw = np.char.replace(body[:, col_idx['company_size']], '/', '-')
    size_map = {
        '<10': 0, '10-49': 1, '50-99': 2, '100-500': 3,
        '500-999': 4, '1000-4999': 5, '5000-9999': 6, '10000+': 7
    }
    processed['company_size'] = map_ordinal(size_raw, size_map)

    # 5. Last New Job
    job_map = {'never': 0, '>4': 5}
    job_map.update({str(i): i for i in range(1, 5)})
    processed['last_new_job'] = map_ordinal(body[:, col_idx['last_new_job']], job_map)

    # 6. Education Level
    edu_map = {'Primary School': 0, 'High School': 1, 'Graduate': 2, 'Masters': 3, 'Phd': 4}
    processed['education_level'] = map_ordinal(body[:, col_idx['education_level']], edu_map)

    # NOMINAL CATEGORICAL FEATURES
    nominal_cols = ['gender', 'relevent_experience', 'enrolled_university', 
                    'major_discipline', 'company_type'] 
    city_col = body[:, col_idx['city']]
    uniques, counts = np.unique(city_col, return_counts=True)
    freq_map = dict(zip(uniques, counts / len(city_col)))
    processed['city_freq'] = np.array([freq_map.get(x, 0) for x in city_col])

    # One-Hot Encoding for others
    for col in nominal_cols:
        raw_col = body[:, col_idx[col]]
        raw_col = np.where(raw_col == '', 'Unknown', raw_col)
        oh_matrix, labels = one_hot(raw_col)
        processed[f'{col}_onehot'] = oh_matrix
    y = body[:, col_idx['target']]
    processed['target'] = y.astype(float).astype(int)
    return processed

# MISSING VALUE IMPUTATION
def impute_missing(data_dict):
    output = data_dict.copy()
    numeric_cols = ['city_development_index', 'training_hours', 
                    'experience', 'company_size', 'last_new_job', 'education_level']
    for col in numeric_cols:
        if col in output:
            output[col] = fill_missing_numerical(output[col], strategy = 'median')
    return output


def prepare_matrices(processed_data):
    # Numeric & Ordinal Features
    feature_list = [
        processed_data['city_development_index'],
        processed_data['training_hours'],
        processed_data['experience'],
        processed_data['company_size'],
        processed_data['last_new_job'],
        processed_data['education_level'],
        processed_data['city_freq']
    ]
    
    X_numeric = np.column_stack(feature_list)
    
    # One-Hot Encoded Features
    oh_features = []
    for key, val in processed_data.items():
        if '_onehot' in key:
            oh_features.append(val)
            
    # Concatenate
    if oh_features:
        X_categorical = np.column_stack(oh_features)
        X = np.column_stack((X_numeric, X_categorical))
    else:
        X = X_numeric
        
    y = processed_data['target']
    return X, y

def export_processed_data(X, y, raw_data, imputed_data, output_dir = "../data/processed"):
    """
    Export preprocessed dataset to:
        - CSV (human-readable with header)
        - NPY (machine-readable for modeling)
    """
    
    # Output directory and file paths
    os.makedirs(output_dir, exist_ok = True)
    output_csv_path = os.path.join(output_dir, "train_preprocessed.csv")
    output_npy_X_path = os.path.join(output_dir, "X_train.npy")
    output_npy_y_path = os.path.join(output_dir, "y_train.npy")

    # Create the final matrix
    # Ensure X and y have compatible dimensions for stacking
    if len(y.shape) == 1:
        y_reshaped = y.reshape(-1, 1)
    else:
        y_reshaped = y
        
    final_matrix = np.column_stack((X, y_reshaped))

    # CSV header - Numeric Part
    header_numeric = [
        'city_development_index',
        'training_hours',
        'experience',
        'company_size',
        'last_new_job',
        'education_level',
        'city_freq'
    ]

    # CSV header - One-Hot Part
    header_raw = raw_data[0]
    col_idx_notebook = {name: i for i, name in enumerate(header_raw)}
    onehot_headers = []


    for key in imputed_data.keys():
        if "_onehot" in key:
            original_col_name = key.replace("_onehot", "")
            raw_col_vals = raw_data[1:, col_idx_notebook[original_col_name]]
            raw_col_vals = np.where(raw_col_vals == '', 'Unknown', raw_col_vals)
            labels = np.unique(raw_col_vals)
            onehot_headers.extend([f"{original_col_name}_{lbl}" for lbl in labels])
    full_header_list = header_numeric + onehot_headers + ["target"]
    full_header_str = ",".join(full_header_list)

    print(f"Header columns: {len(full_header_list)}")
    print(f"Matrix columns: {final_matrix.shape[1]}")
    assert len(full_header_list) == final_matrix.shape[1], \
        f"Mismatch! Header: {len(full_header_list)}, Matrix: {final_matrix.shape[1]}"

    # Save files
    # CSV
    np.savetxt(
        output_csv_path,
        final_matrix,
        delimiter=",",
        header=full_header_str,
        fmt="%.6f",
        comments=""
    )
    print(f"CSV exported to: {output_csv_path}")

    # NPY
    np.save(output_npy_X_path, X)
    np.save(output_npy_y_path, y)
    print(f"NPY files exported to: {output_dir}")
