import pandas as pd 
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
import os
from datetime import datetime
import pickle
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import openpyxl
import xlrd

def clean_text(text):
    if pd.isna(text):
        return text
    text = str(text)
    text = re.sub(r'[^a-zA-Z0-9\s\.,_-]', '', text)
    text = ' '.join(text.split())
    return text.strip()

def read_excel_file(file_path_or_buffer):
    """
    Read various types of Excel files (xls, xlsx, xlsm, etc.)
    
    Parameters:
    - file_path_or_buffer: String path or file buffer
    
    Returns:
    - DataFrame, error_message (if any)
    """
    try:
        # Try reading with default pandas read_excel
        df = pd.read_excel(file_path_or_buffer)
        return df, None
    except Exception as e1:
        try:
            # Try with openpyxl for newer Excel files
            df = pd.read_excel(file_path_or_buffer, engine='openpyxl')
            return df, None
        except Exception as e2:
            try:
                # Try with xlrd for older Excel files
                df = pd.read_excel(file_path_or_buffer, engine='xlrd')
                return df, None
            except Exception as e3:
                return None, f"Could not read file. Errors: \n1. {str(e1)}\n2. {str(e2)}\n3. {str(e3)}"

def identify_column_types(df):
    """
    Identify the type of each column (numerical, date, categorical, text)
    """
    column_types = {}
    
    for column in df.columns:
        sample = df[column].dropna()
        if len(sample) == 0:
            column_types[column] = 'empty'
            continue
       
        if pd.api.types.is_numeric_dtype(df[column]):
            if df[column].nunique() / len(df[column].dropna()) < 0.05:  # If less than 5% unique values
                column_types[column] = 'categorical_numeric'
            else:
                column_types[column] = 'numeric'
            continue
       
        try:
            pd.to_datetime(sample, errors='raise')
            column_types[column] = 'date'
            continue
        except (ValueError, TypeError):
            pass
        
        unique_ratio = df[column].nunique() / len(df[column].dropna())
        if unique_ratio < 0.05:  # If less than 5% unique values
            column_types[column] = 'categorical'
        else:
            # Check average word count
            avg_words = df[column].str.split().str.len().mean()
            if avg_words > 3:
                column_types[column] = 'text'
            else:
                column_types[column] = 'categorical'
    
    return column_types

def perform_eda(df):
    # Add column type identification at the start of EDA
    column_types = identify_column_types(df)
    print("\n=== Column Type Analysis ===")
    for col, col_type in column_types.items():
        print(f"{col}: {col_type}")

    print("\n=== Basic Dataset Information ===")
    print(f"Shape of dataset: {df.shape}")
    print("\nColumn Names:", df.columns.tolist())
    
    print("\n=== Missing Value Analysis ===")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    print("\n=== Data Types ===")
    print(df.dtypes)
    
    print("\n=== Basic Statistics ===")
    print(df.describe(include='all'))
    
    # Identify numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) >= 2:
        print("\n=== Correlation Analysis ===")
        correlation = df[numeric_columns].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.show()
    
    print("\n=== Value Counts for Categorical Columns ===")
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        print(f"\nUnique values in {col}:")
        print(df[col].value_counts().head())

    return {
        'missing_values': missing_values,
        'numeric_columns': numeric_columns,
        'categorical_columns': categorical_columns,
        'shape': df.shape
    }

def normalize_data(df, method='minmax', columns=None):
    """
    Normalize specified columns using either MinMax or Standard scaling
    """
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    df_normalized = df.copy()
    if method == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
        
    df_normalized[columns] = scaler.fit_transform(df_normalized[columns])
    return df_normalized

def impute_missing_values(df, strategy='mean'):
    """
    Impute missing values based on data type and strategy
    """
    df_imputed = df.copy()
    
    # Numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_columns) > 0:
        num_imputer = SimpleImputer(strategy=strategy)
        df_imputed[numeric_columns] = num_imputer.fit_transform(df[numeric_columns])
    
    # Categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        # For categorical, we always use most_frequent
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_imputed[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])
    
    return df_imputed

def prepare_data_for_training(df, target_column):
    """
    Prepare data for model training with proper handling of missing values
    """
    df_encoded = df.copy()
    label_encoders = {}
    
    for column in df_encoded.select_dtypes(include=['object']).columns:
        # Add 'missing' to the set of unique values
        unique_values = df_encoded[column].fillna('missing').unique()
        if 'missing' not in unique_values:
            unique_values = np.append(unique_values, 'missing')
            
        # Create and fit encoder with all possible values including 'missing'
        label_encoders[column] = LabelEncoder()
        label_encoders[column].fit(unique_values)
        
        # Transform the data
        df_encoded[column] = df_encoded[column].fillna('missing')
        df_encoded[column] = label_encoders[column].transform(df_encoded[column])
    
    X = df_encoded.drop(target_column, axis=1)
    y = df_encoded[target_column]
    
    return X, y, label_encoders

def train_decision_tree(X, y, progress_callback=None):
    """
    Train decision tree with progress tracking
    """
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train model
        model = DecisionTreeClassifier(random_state=42)
        
        # Simulate progress steps
        total_steps = 5
        for i in range(total_steps):
            if progress_callback:
                progress_callback((i + 1) / total_steps, f"Training step {i+1}/{total_steps}")
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Convert dict report to string format
        report_str = classification_report(y_test, y_pred, zero_division=0)
        
        return {
            'model': model,
            'accuracy': accuracy,
            'report': report_str,
            'report_dict': report,  # Adding dictionary version for easier parsing
            'test_data': (X_test, y_test)
        }
    except Exception as e:
        print(f"Error in train_decision_tree: {str(e)}")
        raise

def train_svm(X, y, progress_callback=None):
    """
    Train SVM model with progress tracking
    """
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train model
        model = SVC(kernel='rbf', probability=True, random_state=42)
        
        # Simulate progress steps
        total_steps = 5
        for i in range(total_steps):
            if progress_callback:
                progress_callback((i + 1) / total_steps, f"Training SVM step {i+1}/{total_steps}")
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        return {
            'model': model,
            'accuracy': accuracy,
            'report': classification_report(y_test, y_pred, zero_division=0),
            'report_dict': report,
            'test_data': (X_test, y_test)
        }
    except Exception as e:
        print(f"Error in train_svm: {str(e)}")
        raise

def train_random_forest(X, y, progress_callback=None):
    """Train Random Forest model with progress tracking"""
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        total_steps = 5
        for i in range(total_steps):
            if progress_callback:
                progress_callback((i + 1) / total_steps, f"Training Random Forest step {i+1}/{total_steps}")
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        return {
            'model': model,
            'accuracy': accuracy,
            'report': classification_report(y_test, y_pred, zero_division=0),
            'report_dict': report,
            'feature_importance': pd.Series(model.feature_importances_, index=X.columns),
            'test_data': (X_test, y_test)
        }
    except Exception as e:
        print(f"Error in train_random_forest: {str(e)}")
        raise

def train_xgboost(X, y, progress_callback=None):
    """Train XGBoost model with progress tracking"""
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = xgb.XGBClassifier(random_state=42)
        
        total_steps = 5
        for i in range(total_steps):
            if progress_callback:
                progress_callback((i + 1) / total_steps, f"Training XGBoost step {i+1}/{total_steps}")
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        return {
            'model': model,
            'accuracy': accuracy,
            'report': classification_report(y_test, y_pred, zero_division=0),
            'report_dict': report,
            'feature_importance': pd.Series(model.feature_importances_, index=X.columns),
            'test_data': (X_test, y_test)
        }
    except Exception as e:
        print(f"Error in train_xgboost: {str(e)}")
        raise

def train_ensemble(X, y, models_to_use=['decision_tree', 'random_forest', 'svm', 'xgboost'], progress_callback=None):
    """Train ensemble model using voting classifier"""
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize individual models
        model_dict = {
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42)
        }
        
        # Select requested models
        estimators = [(name, model_dict[name]) for name in models_to_use]
        
        # Create and train voting classifier
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        
        total_steps = len(models_to_use) + 1
        for i, model_name in enumerate(models_to_use):
            if progress_callback:
                progress_callback((i + 1) / total_steps, f"Training {model_name}")
        
        ensemble.fit(X_train, y_train)
        
        # Final evaluation
        if progress_callback:
            progress_callback(1.0, "Evaluating ensemble")
        
        y_pred = ensemble.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Cross-validation score
        cv_scores = cross_val_score(ensemble, X, y, cv=5)
        
        return {
            'model': ensemble,
            'accuracy': accuracy,
            'report': classification_report(y_test, y_pred, zero_division=0),
            'report_dict': report,
            'cv_scores': cv_scores,
            'test_data': (X_test, y_test)
        }
    except Exception as e:
        print(f"Error in train_ensemble: {str(e)}")
        raise

def train_neural_network(X, y, task_type='classification', optimizer='adam', loss=None, 
                        epochs=50, batch_size=32, hidden_activation='relu', 
                        output_activation=None, progress_callback=None):
    """Train a neural network and return standardized results"""
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if task_type == 'classification':
            y_train = to_categorical(y_train)
            y_test = to_categorical(y_test)
            output_units = y_train.shape[1]
            if loss is None:
                loss = 'categorical_crossentropy'
            metrics = ['accuracy']
            if output_activation is None:
                output_activation = 'softmax'
        else:
            output_units = 1
            if loss is None:
                loss = 'mean_squared_error'
            metrics = ['mae']
            if output_activation is None:
                output_activation = 'linear'
        
        model = Sequential([
            Dense(64, activation=hidden_activation, input_shape=(X.shape[1],)),
            Dense(32, activation=hidden_activation),
            Dense(output_units, activation=output_activation)
        ])
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        # Train with progress tracking
        total_steps = epochs
        for epoch in range(epochs):
            if progress_callback:
                progress_callback((epoch + 1) / total_steps, f"Training epoch {epoch+1}/{total_steps}")
            model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=0)
        
        # Generate predictions and metrics
        y_pred = model.predict(X_test)
        if task_type == 'classification':
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_test_classes = np.argmax(y_test, axis=1)
            accuracy = accuracy_score(y_test_classes, y_pred_classes)
            report = classification_report(y_test_classes, y_pred_classes, output_dict=True, zero_division=0)
            report_str = classification_report(y_test_classes, y_pred_classes, zero_division=0)
        else:
            accuracy = model.evaluate(X_test, y_test, verbose=0)[1]  # Use MAE for regression
            report = {'mae': accuracy}
            report_str = f"Mean Absolute Error: {accuracy}"
        
        return {
            'model': model,
            'accuracy': accuracy,
            'report': report_str,
            'report_dict': report,
            'test_data': (X_test, y_test),
            'history': model.history.history
        }
    
    except Exception as e:
        print(f"Error in train_neural_network: {str(e)}")
        raise

def save_model_history(results, target_column, model_type='decision_tree'):
    """Save model training results to JSON and model to pickle"""
    history_file = "model_history.json"
    model_file = f"models/{model_type}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save model and encoders in pickle format
    model_data = {
        'model': results['model'],
        'target_column': target_column,
        'accuracy': results['accuracy'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'report_dict': results['report_dict']
    }
    
    # Save using pickle
    with open(model_file, 'wb') as f:
        pickle.dump(model_data, f)
    
    # Prepare history entry
    history_entry = {
        'timestamp': model_data['timestamp'],
        'target_column': target_column,
        'model_type': model_type,
        'accuracy': float(results['accuracy']),
        'model_file': model_file,
        'report': results['report']
    }
    
    # Load existing history
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = []
    
    # Add new entry and keep only last 5
    history.append(history_entry)
    history = history[-5:]
    
    # Save updated history
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4)
    
    return history

def load_model_history():
    """Load model training history with backward compatibility"""
    history_file = "model_history.json"
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
            # Add model_type for legacy entries
            for entry in history:
                if 'model_type' not in entry:
                    entry['model_type'] = 'decision_tree'  # default for old entries
            return history
    return []

def evaluate_model_with_test_data(model_file, test_data, encoders=None):
    """Evaluate saved model with new test data"""
    # Load model from pickle file
    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    
    # Prepare test data using the same encoders if provided
    if encoders:
        for column in test_data.select_dtypes(include=['object']).columns:
            if column in encoders:
                test_data[column] = encoders[column].transform(test_data[column].fillna('missing'))
    
    predictions = model.predict(test_data)
    return predictions, model_data['accuracy']

def convert_column_type(df, column_name, target_type, date_format=None):
    """
    Convert column to specified type with error handling
    """
    df = df.copy()
    try:
        if target_type == 'numeric':
            # Remove non-numeric characters except decimal point and negative sign
            df[column_name] = df[column_name].str.replace(r'[^\d.-]', '', regex=True)
            df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
            
        elif target_type == 'date':
            if date_format:
                df[column_name] = pd.to_datetime(df[column_name], format=date_format, errors='coerce')
            else:
                df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
                
        elif target_type == 'string':
            df[column_name] = df[column_name].astype(str)
            
        return df, None  # None means no error
        
    except Exception as e:
        return df, str(e)

def suggest_date_format(sample_value):
    """
    Suggest possible date formats for a given sample value
    """
    common_formats = [
        '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y',
        '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y',
        '%Y%m%d', '%d%m%Y', '%m%d%Y',
        '%d-%b-%Y', '%Y-%b-%d',
        '%d/%b/%Y', '%Y/%b/%d'
    ]
    
    valid_formats = []
    for date_format in common_formats:
        try:
            pd.to_datetime(sample_value, format=date_format)
            valid_formats.append(date_format)
        except:
            continue
    
    return valid_formats

def predict_single_column(model_data, input_column, encoders=None):
    """Make predictions for a single column using loaded model"""
    try:
        model = model_data['model']
        if encoders and isinstance(input_column, pd.Series):
            # Convert to DataFrame to maintain column name
            input_df = pd.DataFrame({input_column.name: input_column})
            # Apply encoding if needed
            if input_column.name in encoders:
                input_df[input_column.name] = encoders[input_column.name].transform(input_df[input_column.name].fillna('missing'))
            return model.predict(input_df), None
    except Exception as e:
        return None, str(e)

def perform_pca(df, n_components=None, variance_threshold=0.95):
    """
    Perform PCA on numeric columns
    
    Parameters:
    - df: DataFrame
    - n_components: int or None (if None, use variance threshold)
    - variance_threshold: float (used if n_components is None)
    
    Returns:
    - transformed data
    - PCA object
    - explained variance ratios
    - feature names
    """
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) == 0:
        return None, None, None, None
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_cols])
    
    # Determine number of components
    if n_components is None:
        n_components = min(len(numeric_cols), len(df))
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    
    # Calculate cumulative variance ratio
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    
    # If using variance threshold, determine number of components
    if n_components is None:
        n_components = np.argmax(cumulative_variance_ratio >= variance_threshold) + 1
        pca_result = pca_result[:, :n_components]
    
    # Create feature names
    feature_names = [f'PC{i+1}' for i in range(n_components)]
    
    return pd.DataFrame(pca_result, columns=feature_names), pca, pca.explained_variance_ratio_, numeric_cols

def select_k_best_features(df, target_column, k=10):
    """
    Select K best features using chi-squared test
    
    Parameters:
    - df: DataFrame
    - target_column: str
    - k: int (number of top features to select)
    
    Returns:
    - DataFrame with selected features
    - List of selected feature names
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Apply SelectKBest
    selector = SelectKBest(score_func=chi2, k=k)
    X_new = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()]
    
    return pd.DataFrame(X_new, columns=selected_features), selected_features

def predict_single_value(model_data, input_values, encoders=None):
    """
    Make prediction for a single input with handling for unseen categories
    """
    try:
        # Create a single-row DataFrame with the same columns
        input_df = pd.DataFrame([input_values])
        
        # Apply preprocessing with handling for unseen categories
        if encoders:
            for column in input_df.columns:
                if column in encoders:
                    encoder = encoders[column]
                    value = input_df[column].iloc[0]
                    
                    # Handle empty or missing values
                    if pd.isna(value) or value == '':
                        value = 'missing'
                    
                    # Handle unseen categories
                    try:
                        classes = encoder.classes_
                        if value not in classes:
                            value = 'missing'
                            print(f"Warning: Unseen category in column {column}. Using 'missing' as default.")
                    except:
                        value = 'missing'
                    
                    # Ensure 'missing' is in encoder classes
                    if 'missing' not in encoder.classes_:
                        # Add 'missing' to encoder classes
                        encoder.classes_ = np.append(encoder.classes_, ['missing'])
                    
                    # Transform the value
                    input_df[column] = value
                    input_df[column] = encoder.transform(input_df[column])
        
        # Make prediction
        model = model_data['model']
        prediction = model.predict(input_df)
        
        # Get probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(input_df)
            
        # Format result
        result = {
            'prediction': prediction[0],
            'probability': probability[0] if probability is not None else None,
            'input_values': input_values
        }
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise Exception(f"Prediction error: {str(e)}")

