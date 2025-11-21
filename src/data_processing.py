import numpy as np

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