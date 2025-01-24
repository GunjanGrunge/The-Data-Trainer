import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import pickle  
import seaborn as sns
from main import (clean_text, perform_eda, normalize_data, impute_missing_values,
                 prepare_data_for_training, train_decision_tree, save_model_history,
                 load_model_history, evaluate_model_with_test_data, train_svm,
                 identify_column_types, convert_column_type, suggest_date_format,
                 train_random_forest, train_xgboost, train_ensemble,
                 predict_single_column, perform_pca, select_k_best_features,
                 train_neural_network, read_excel_file, predict_single_value)  # Added predict_single_value

# Set page config first
st.set_page_config(page_title="The Data Trainer", layout="wide")

# Initialize session state with all required variables
if 'df' not in st.session_state:
    st.session_state.df = None  # Initialize as None instead of dummy_df
if 'data_types' not in st.session_state:
    st.session_state.data_types = {}  # Initialize as empty dict
if 'conversions_applied' not in st.session_state:
    st.session_state.conversions_applied = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Convert Columns"
if 'conversion_status' not in st.session_state:
    st.session_state.conversion_status = None
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'last_trained_encoders' not in st.session_state:  # Add this line
    st.session_state.last_trained_encoders = None

def update_session_state(new_df, message=None):
    """Update all session state variables and data types"""
    st.session_state.df = new_df.copy()
    st.session_state.data_types = {col: str(new_df[col].dtype) for col in new_df.columns}
    st.session_state.conversions_applied = True
    if message:
        st.session_state.conversion_status = message
    # Force numeric columns identification
    st.session_state.numeric_columns = new_df.select_dtypes(
        include=['float64', 'int64', 'int32', 'float32']
    ).columns.tolist()

def handle_column_conversion(df, column_to_convert, target_type, date_format=None):
    """Handle column conversion and state updates"""
    new_df, error = convert_column_type(df, column_to_convert, target_type, date_format)
    
    if not error:
        # Force correct data type
        try:
            if target_type == "numeric":
                new_df[column_to_convert] = pd.to_numeric(new_df[column_to_convert], errors='coerce')
            elif target_type == "date":
                new_df[column_to_convert] = pd.to_datetime(new_df[column_to_convert], errors='coerce')
            elif target_type == "string":
                new_df[column_to_convert] = new_df[column_to_convert].astype(str)
            
            # Update session state with new dataframe
            update_session_state(
                new_df, 
                f"Column '{column_to_convert}' successfully converted to {target_type}"
            )
            return True
        except Exception as e:
            st.error(f"Error forcing data type: {str(e)}")
            return False
    else:
        st.error(f"Error during conversion: {error}")
        return False

# Add this function near the top of the file, after initializing session state
def update_progress(progress, text):
    """Update progress bar and text in Streamlit"""
    if 'progress_bar' in st.session_state:
        st.session_state.progress_bar.progress(progress)
    if 'progress_text' in st.session_state:
        st.session_state.progress_text.text(text)

# Add in sidebar at the top
if st.sidebar.button('🏠 Home', key='home_btn'):
    st.session_state.page = 'home'
    st.rerun()

# Display current data types in sidebar
st.sidebar.write("### Current Data Types")
current_types = pd.DataFrame(
    [(col, dtype) for col, dtype in st.session_state.data_types.items()],
    columns=['Column', 'Type']
)
st.sidebar.dataframe(current_types, height=150)

# Add this after the current data types display in sidebar
st.sidebar.write("---")  # Add a separator
if st.sidebar.button("📊 Compare Models", key="compare_models_btn"):
    st.subheader("Model Performance Comparison")
    
    # Load model history
    history = load_model_history()
    if history:
        # Create DataFrame for comparison
        comparison_df = pd.DataFrame([
            {
                'Model Type': entry['model_type'],
                'Target': entry['target_column'],
                'Accuracy': entry['accuracy'],
                'Timestamp': entry['timestamp']
            }
            for entry in history
        ])
        
        # Display overall comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Model Performance Summary")
            summary_stats = comparison_df.groupby('Model Type')['Accuracy'].agg(['mean', 'std', 'count'])
            summary_stats.columns = ['Average Accuracy', 'Std Dev', 'Count']
            st.dataframe(summary_stats.round(4))
            
            # Plot average performance by model type
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=comparison_df, x='Model Type', y='Accuracy')
            plt.xticks(rotation=45)
            plt.title('Average Model Performance by Type')
            st.pyplot(fig)
        
        with col2:
            st.write("### Performance Trend Over Time")
            fig, ax = plt.subplots(figsize=(10, 6))
            for model_type in comparison_df['Model Type'].unique():
                model_data = comparison_df[comparison_df['Model Type'] == model_type]
                plt.plot(model_data['Timestamp'], model_data['Accuracy'], 
                        marker='o', label=model_type)
            plt.xticks(rotation=45)
            plt.legend()
            plt.title('Model Performance Over Time')
            st.pyplot(fig)
        
        # Detailed comparison table
        st.write("### Detailed Model Comparison")
        st.dataframe(comparison_df.sort_values('Timestamp', ascending=False))
        
        # Target-specific analysis
        st.write("### Performance by Target Column")
        target_stats = comparison_df.groupby(['Target', 'Model Type'])['Accuracy'].mean().unstack()
        st.dataframe(target_stats.round(4))
        
        # Statistical analysis
        st.write("### Statistical Analysis")
        if len(comparison_df['Model Type'].unique()) > 1:
            try:
                from scipy import stats
                model_accuracies = {model: comparison_df[comparison_df['Model Type'] == model]['Accuracy']
                                  for model in comparison_df['Model Type'].unique()}
                
                f_stat, p_val = stats.f_oneway(*model_accuracies.values())
                st.write(f"One-way ANOVA test p-value: {p_val:.4f}")
                if p_val < 0.05:
                    st.write("There is a statistically significant difference between model performances.")
                else:
                    st.write("No statistically significant difference between model performances.")
            except:
                st.write("Could not perform statistical analysis due to insufficient data.")
    else:
        st.info("No model history available for comparison.")

if st.session_state.page == 'home':
    st.title("Column Mapping Analysis")
    st.write("### Upload your Excel file to begin analysis")
    
    uploaded_file = st.file_uploader(
        "Drag and drop your Excel file here",
        type=['xlsx', 'xls', 'xlsm'],
        help="Supported formats: .xlsx, .xls, .xlsm"
    )
    
    if uploaded_file is not None:
        try:
            with st.spinner("Reading file..."):
                df, error = read_excel_file(uploaded_file)
                
            if error:
                st.error(error)
            else:
                st.success("File uploaded successfully!")
                st.write("### Preview of uploaded data")
                st.dataframe(df.head())
                
                if st.button("Proceed to Analysis", key="proceed_btn"):
                    st.session_state.df = df.copy()
                    st.session_state.data_types = {col: str(df[col].dtype) for col in df.columns}
                    st.session_state.page = 'analysis'
                    st.rerun()
                
                # Show basic information
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### Data Shape")
                    st.write(f"Rows: {df.shape[0]}")
                    st.write(f"Columns: {df.shape[1]}")
                
                with col2:
                    st.write("### Column Types")
                    type_counts = df.dtypes.value_counts()
                    st.write(type_counts)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please make sure your file is a valid Excel file.")
    
    # Add some instructions
    with st.expander("Instructions"):
        st.write("""
        1. Upload your Excel file using the drag & drop area above
        2. Preview the data to ensure it was read correctly
        3. Click 'Proceed to Analysis' to begin working with your data
        4. Use the sidebar options to perform various analyses
        """)

else:
    st.title("Column Mapping Data Analysis Dashboard")

    # Get current dataframe from session state
    df = st.session_state.df

    # Show current data preview
    st.subheader("Current Data Preview")
    st.dataframe(df.head())

    # Update processing options to include Feature Selection
    processing_options = st.sidebar.radio(
        "Select Processing Step:",
        ["Convert Columns", "Clean Data", "Handle Missing Values", "Normalize Data", "PCA", "Feature Selection", "Train Model", "View EDA"],
        key="processing_radio",
        on_change=lambda: setattr(st.session_state, 'current_page', st.session_state.processing_radio)
    )

    if processing_options == "Clean Data":
        if st.button("Clean Data", key="clean_data_btn"):
            for column in df.columns:
                df[column] = df[column].apply(clean_text)
            st.subheader("Cleaned Data")
            st.dataframe(df)

    elif processing_options == "Handle Missing Values":
        numeric_imputation = st.selectbox(
            "Choose imputation strategy for numeric data:",
            ["mean", "median", "most_frequent"]
        )
        
        if st.button("Impute Missing Values", key="impute_missing_btn"):
            df = impute_missing_values(df, strategy=numeric_imputation)
            st.write("Data after imputation:")
            st.dataframe(df)

    elif processing_options == "Normalize Data":
        st.subheader("Data Normalization")
        
        # Get current numeric columns from session state
        numeric_columns = st.session_state.df.select_dtypes(
            include=['float64', 'int64', 'int32', 'float32']
        ).columns.tolist()
        
        if len(numeric_columns) == 0:
            st.warning("No numeric columns detected. Please convert columns to numeric type first.")
            if st.button("Go to Column Conversion"):
                st.session_state.current_page = "Convert Columns"
                st.rerun()
        else:
            normalize_method = st.selectbox(
                "Choose normalization method:",
                ["minmax", "standard"]
            )
            
            # Fixed syntax error in multiselect
            columns_to_normalize = st.multiselect(
                label="Select columns to normalize:",
                options=numeric_columns,
                default=[numeric_columns[0]] if numeric_columns else None
            )
            
            if st.button("Normalize Data", key="normalize_data_btn"):
                if len(columns_to_normalize) > 0:
                    df = normalize_data(df, method=normalize_method, columns=columns_to_normalize)
                    st.write("Normalized Data:")
                    st.dataframe(df)
                else:
                    st.warning("Please select columns to normalize")

    elif processing_options == "Convert Columns":
        st.subheader("Convert Column Types")
        
        col1, col2 = st.columns(2)
        
        with col1:
            column_to_convert = st.selectbox(
                "Select column to convert",
                st.session_state.df.columns.tolist()
            )
            st.write(f"Current type: {st.session_state.data_types.get(column_to_convert, 'Unknown')}")
            
        with col2:
            target_type = st.selectbox(
                "Convert to",
                ["numeric", "date", "string"]
            )
        
        # Show date format input if converting to date
        date_format = None
        if target_type == "date":
            sample_value = df[column_to_convert].iloc[0]
            suggested_formats = suggest_date_format(sample_value)
            
            st.write(f"Sample value: {sample_value}")
            if suggested_formats:
                date_format = st.selectbox(
                    "Select date format",
                    suggested_formats,
                    help="Choose the format that matches your data"
                )
            else:
                date_format = st.text_input(
                    "Enter date format",
                    help="e.g., %Y-%m-%d for YYYY-MM-DD"
                )
        
        if st.button("Convert Column", key="convert_column_btn"):
            if handle_column_conversion(st.session_state.df, column_to_convert, target_type, date_format):
                st.rerun()

    elif processing_options == "Train Model":
        st.subheader("Train Model")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Train New Model", "Ensemble Training", "Model History", "Test Model"])
        
        with tab1:
            model_type = st.selectbox(
                "Select Model Type",
                ["Decision Tree", "Random Forest", "SVM", "XGBoost", "Neural Network"],
                help="Choose the type of model to train"
            )
            
            target_column = st.selectbox("Select target column for prediction:", df.columns.tolist())
            
            # Model specific parameters
            if model_type == "SVM":
                kernel = st.selectbox(
                    "Select SVM Kernel",
                    ["rbf", "linear", "poly", "sigmoid"]
                )
            elif model_type in ["Random Forest", "XGBoost"]:
                n_estimators = st.slider("Number of estimators", 50, 500, 100)
            elif model_type == "Neural Network":
                task_type = st.selectbox("Select task type:", ["classification", "regression"])
                
                # Add activation function selection
                activation_functions = ["relu", "tanh", "sigmoid", "linear", "elu", "selu"]
                hidden_activation = st.selectbox(
                    "Select hidden layer activation function:",
                    activation_functions,
                    index=0,  # relu as default
                    help="Activation function for hidden layers"
                )
                
                output_activation = None
                if task_type == "classification":
                    output_activation = st.selectbox(
                        "Select output layer activation function:",
                        ["softmax", "sigmoid"],
                        index=0,
                        help="Activation function for output layer"
                    )
                else:
                    output_activation = "linear"  # for regression tasks
                
                optimizer = st.selectbox("Select optimizer:", ["adam", "sgd", "rmsprop"])
                loss = st.text_input("Enter loss function (leave blank for default):")
                epochs = st.slider("Number of epochs", 10, 100, 50)
                batch_size = st.slider("Batch size", 16, 128, 32)
                
                if st.button("Train Neural Network", key="train_neural_network_btn"):
                    try:
                        # Create progress tracking elements
                        st.session_state.progress_bar = st.progress(0)
                        st.session_state.progress_text = st.empty()
                        
                        X, y, encoders = prepare_data_for_training(df, target_column)
                        results = train_neural_network(
                            X, y,
                            task_type=task_type,
                            optimizer=optimizer,
                            loss=loss if loss else None,
                            epochs=epochs,
                            batch_size=batch_size,
                            hidden_activation=hidden_activation,
                            output_activation=output_activation,
                            progress_callback=update_progress
                        )
                        st.session_state.last_trained_encoders = encoders
                        
                        # Display results
                        st.success(f"Training completed! Accuracy: {results['accuracy']:.4f}")
                        st.write("Model Report:")
                        st.text(results['report'])
                        
                        # Save model
                        save_model_history(results, target_column, 'neural_network')
                        
                    except Exception as e:
                        st.error(f"Error during training: {str(e)}")

            def train_and_display_model():
                with st.spinner("Preparing data..."):
                    X, y, encoders = prepare_data_for_training(df, target_column)
                
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                def update_progress(progress, text):
                    progress_bar.progress(progress)
                    progress_text.text(text)
                
                # Train selected model
                if model_type == "Decision Tree":
                    results = train_decision_tree(X, y, update_progress)
                    model_key = 'decision_tree'
                elif model_type == "Random Forest":
                    results = train_random_forest(X, y, update_progress)
                    model_key = 'random_forest'
                elif model_type == "XGBoost":
                    results = train_xgboost(X, y, update_progress)
                    model_key = 'xgboost'
                elif model_type == "Neural Network":
                    model, history = train_neural_network(X, y, task_type, optimizer, loss if loss else None, epochs, batch_size, update_progress)
                    results = {
                        'model': model,
                        'accuracy': history['accuracy'][-1] if task_type == 'classification' else history['mae'][-1],
                        'report_dict': history,
                        'history': history
                    }
                    model_key = 'neural_network'
                else:
                    results = train_svm(X, y, update_progress)
                    model_key = 'svm'
                
                # Save and display results
                history = save_model_history(results, target_column, model_key)
                
                # Display results including feature importance if available
                st.success(f"Training completed for {model_type}!")
                st.write("Model Accuracy:", f"{results['accuracy']:.4f}")
                
                if 'feature_importance' in results:
                    st.write("### Feature Importance")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    results['feature_importance'].sort_values().plot(kind='barh')
                    plt.title('Feature Importance')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Display classification report for non-neural network models
                if model_type != "Neural Network":
                    report_dict = results['report_dict']
                    display_classification_report(report_dict)
                
                return results, encoders

            def display_classification_report(report_dict):
                # Create main metrics dataframe
                metrics_data = []
                for class_label, metrics in report_dict.items():
                    if class_label not in ['accuracy', 'macro avg', 'weighted avg']:
                        metrics_data.append({
                            'Class': class_label,
                            'Precision': metrics['precision'],
                            'Recall': metrics['recall'],
                            'F1-Score': metrics['f1-score'],
                            'Support': metrics['support']
                        })
                
                if metrics_data:
                    report_df = pd.DataFrame(metrics_data)
                    st.write("Classification Report - Class Metrics:")
                    st.dataframe(report_df, use_container_width=True)
                    
                    # Create and display visualization
                    fig, ax = plt.subplots(figsize=(10, 6))
                    report_df.plot(x='Class', y=['Precision', 'Recall', 'F1-Score'], 
                                 kind='bar', ax=ax)
                    plt.title('Model Performance Metrics by Class')
                    plt.tight_layout()
                    st.pyplot(fig)
            
            if st.button("Train Model", key="train_model_btn"):
                try:
                    # Create progress tracking elements
                    st.session_state.progress_bar = st.progress(0)
                    st.session_state.progress_text = st.empty()
                    
                    results, encoders = train_and_display_model()
                    st.session_state.last_trained_encoders = encoders
                except Exception as e:
                    st.error(f"An error occurred during training: {str(e)}")

        with tab2:
            st.subheader("Ensemble Training")
            target_column = st.selectbox(
                "Select target column for ensemble prediction:", 
                df.columns.tolist(),
                key="ensemble_target"
            )
            
            models_to_use = st.multiselect(
                "Select models for ensemble",
                ["decision_tree", "random_forest", "svm", "xgboost"],
                default=["decision_tree", "random_forest", "svm"]
            )
            
            if st.button("Train Ensemble", key="train_ensemble_btn"):
                try:
                    # Create progress tracking elements
                    st.session_state.progress_bar = st.progress(0)
                    st.session_state.progress_text = st.empty()
                    
                    with st.spinner("Preparing data..."):
                        X, y, encoders = prepare_data_for_training(df, target_column)
                    
                    progress_bar = st.progress(0)
                    progress_text = st.empty()
                    
                    def update_progress(progress, text):
                        progress_bar.progress(progress)
                        progress_text.text(text)
                    
                    # Train ensemble with progress tracking
                    results = train_ensemble(X, y, models_to_use, update_progress)
                    st.session_state.last_trained_encoders = encoders
                    
                    history = save_model_history(results, target_column, 'ensemble')
                    
                    st.success("Ensemble Training Completed!")
                    st.write("Ensemble Accuracy:", f"{results['accuracy']:.4f}")
                    
                    # Display cross-validation scores
                    st.write("### Cross-validation Scores")
                    cv_df = pd.DataFrame({
                        'Fold': range(1, 6),
                        'Accuracy': results['cv_scores']
                    })
                    st.dataframe(cv_df)
                    
                    # Show mean and std of CV scores
                    st.write(f"Mean CV Accuracy: {results['cv_scores'].mean():.4f} (±{results['cv_scores'].std():.4f})")
                    
                    # Display classification report
                    display_classification_report(results['report_dict'])
                    
                except Exception as e:
                    st.error(f"An error occurred during ensemble training: {str(e)}")
                    st.error("Full error:", str(e.__class__.__name__), str(e))

        with tab3:
            st.subheader("Model Training History")
            history = load_model_history()
            if history:
                # Get unique model types with error handling
                model_types = list(set(entry.get('model_type', 'Unknown') for entry in history))
                
                # Add model type filter
                model_filter = st.multiselect(
                    "Filter by Model Type",
                    options=model_types,
                    default=model_types
                )
                
                # Filter history based on selection with error handling
                filtered_history = [
                    entry for entry in history 
                    if entry.get('model_type', 'Unknown') in model_filter
                ]
                
                if filtered_history:
                    # Create comparison chart
                    history_df = pd.DataFrame(filtered_history)
                    
                    # Group by model type for comparison
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for model_type in model_filter:
                        model_data = history_df[history_df['model_type'] == model_type]
                        if not model_data.empty:
                            plt.plot(model_data['timestamp'], model_data['accuracy'], 
                                    marker='o', label=model_type)
                    
                    plt.title('Model Accuracy Comparison')
                    plt.xticks(rotation=45)
                    plt.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Show detailed history
                    for entry in filtered_history:
                        model_type = entry.get('model_type', 'Unknown')
                        with st.expander(f"{model_type} trained on {entry['timestamp']}"):
                            st.write(f"Target Column: {entry['target_column']}")
                            st.write(f"Model Type: {model_type}")
                            st.write(f"Accuracy: {entry['accuracy']:.4f}")
                            st.write(f"Model File: {entry['model_file']}")
                else:
                    st.info("No models found for selected types")
            else:
                st.info("No training history available")
        
        with tab4:
            st.subheader("Test Saved Model")
            history = load_model_history()
            if history:
                try:
                    # Add tabs for different testing methods
                    test_tab1, test_tab2 = st.tabs(["Single Value Prediction", "Batch Prediction"])
                    
                    with test_tab1:
                        model_options = [
                            (
                                entry['timestamp'],
                                entry['model_file'],
                                entry['target_column'],
                                f"{entry['model_type']} (Target: {entry['target_column']}, Acc: {entry['accuracy']:.2f})"
                            )
                            for entry in history
                        ]
                        
                        selected_model = st.selectbox(
                            "Select model:",
                            options=model_options,
                            format_func=lambda x: x[3],
                            key="single_model_selector"
                        )
                        
                        if selected_model:
                            try:
                                # Load the model
                                with open(selected_model[1], 'rb') as f:
                                    model_data = pickle.load(f)
                                
                                # Get required columns
                                required_columns = [col for col in st.session_state.df.columns 
                                                 if col != selected_model[2]]  # Exclude target column
                                
                                # Create input fields
                                st.write("### Enter Values for Prediction")
                                input_values = {}
                                
                                # Create columns for better layout
                                cols = st.columns(3)
                                for idx, col in enumerate(required_columns):
                                    with cols[idx % 3]:
                                        if st.session_state.df[col].dtype in ['int64', 'float64']:
                                            input_values[col] = st.number_input(f"Enter {col}", value=0)
                                        else:
                                            input_values[col] = st.text_input(f"Enter {col}", "")
                                
                                if st.button("Predict", key="single_predict_btn"):
                                    try:
                                        result = predict_single_value(
                                            model_data, 
                                            input_values,
                                            st.session_state.last_trained_encoders
                                        )
                                        
                                        # Display results
                                        st.success("Prediction Complete!")
                                        
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write("### Prediction Result")
                                            st.write(f"Predicted {selected_model[2]}: {result['prediction']}")
                                            
                                            if result['probability'] is not None:
                                                st.write("### Prediction Probabilities")
                                                probs_df = pd.DataFrame({
                                                    'Class': range(len(result['probability'])),
                                                    'Probability': result['probability']
                                                })
                                                st.dataframe(probs_df)
                                        
                                        with col2:
                                            st.write("### Input Values")
                                            input_df = pd.DataFrame([input_values])
                                            st.dataframe(input_df)
                                            
                                    except Exception as e:
                                        st.error(f"Prediction error: {str(e)}")
                            except Exception as e:
                                st.error(f"Error loading model: {str(e)}")
                    
                    with test_tab2:
                        # Existing batch prediction code
                        model_options = [
                            (
                                entry['timestamp'],
                                entry['model_file'],
                                f"{entry['model_type']} (Target: {entry['target_column']}, Acc: {entry['accuracy']:.2f})"
                            )
                            for entry in history
                        ]
                        selected_model = st.selectbox(
                            "Select model to test:",
                            options=model_options,
                            format_func=lambda x: x[2],
                            key="batch_model_selector"
                        )
                        
                        if selected_model:
                            # Upload test data
                            test_file = st.file_uploader("Upload test dataset (Excel file)", type="xlsx")
                            if test_file:
                                test_df = pd.read_excel(test_file)
                                st.write("Test Dataset Preview:")
                                st.dataframe(test_df.head())
                                
                                if st.button("Run Test", key="run_test_btn"):
                                    predictions, model_accuracy = evaluate_model_with_test_data(
                                        selected_model[1],
                                        test_df,
                                        st.session_state.encoders
                                    )
                                    st.write("Model Information:")
                                    st.write(f"Original Training Accuracy: {model_accuracy:.4f}")
                                    st.write("Predictions:")
                                    predictions_df = pd.DataFrame(predictions, columns=['Predicted'])
                                    st.dataframe(predictions_df)
                                    
                                    # Show prediction distribution
                                    st.write("### Prediction Distribution")
                                    fig, ax = plt.subplots(figsize=(8, 5))
                                    predictions_df['Predicted'].value_counts().plot(kind='bar')
                                    plt.title('Distribution of Predictions')
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    
                except Exception as e:
                    st.error(f"Error in model selection: {str(e)}")
            else:
                st.info("No saved models available for testing")

    elif processing_options == "PCA":
        st.subheader("Principal Component Analysis")
        
        # Check for numeric columns
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_columns) < 2:
            st.warning("PCA requires at least 2 numeric columns. Please convert or normalize your data first.")
            if st.button("Go to Column Conversion"):
                st.session_state.current_page = "Convert Columns"
                st.rerun()
        else:
            # PCA Options
            pca_method = st.radio(
                "Choose PCA method:",
                ["Number of Components", "Variance Threshold"]
            )
            
            if pca_method == "Number of Components":
                n_components = st.slider(
                    "Select number of components",
                    min_value=1,
                    max_value=max(2, len(numeric_columns)),  # Ensure max_value is at least 2
                    value=min(3, len(numeric_columns))
                )
                variance_threshold = None
            else:
                n_components = None
                variance_threshold = st.slider(
                    "Select variance threshold",
                    min_value=0.1,
                    max_value=0.99,
                    value=0.95,
                    step=0.05
                )
            
            if st.button("Perform PCA", key="perform_pca_btn"):
                try:
                    # Perform PCA
                    pca_df, pca_obj, explained_variance, feature_names = perform_pca(
                        df, n_components, variance_threshold
                    )
                    
                    if pca_df is not None:
                        # Display results
                        st.success("PCA completed successfully!")
                        
                        # Show explained variance
                        st.write("### Explained Variance Ratio")
                        cumulative_variance = np.cumsum(explained_variance)
                        variance_df = pd.DataFrame({
                            'Principal Component': [f'PC{i+1}' for i in range(len(explained_variance))],
                            'Explained Variance Ratio': explained_variance,
                            'Cumulative Variance Ratio': cumulative_variance
                        })
                        st.dataframe(variance_df)
                        
                        # Plot explained variance
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                        
                        # Scree plot
                        ax1.plot(range(1, len(explained_variance) + 1), explained_variance, 'bo-')
                        ax1.set_xlabel('Principal Component')
                        ax1.set_ylabel('Explained Variance Ratio')
                        ax1.set_title('Scree Plot')
                        
                        # Cumulative variance plot
                        ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
                        ax2.set_xlabel('Number of Components')
                        ax2.set_ylabel('Cumulative Explained Variance Ratio')
                        ax2.setTitle('Cumulative Variance Plot')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Show transformed data
                        st.write("### Transformed Data")
                        st.dataframe(pca_df)
                        
                        # Option to save transformed data
                        if st.button("Keep PCA Transformation", key="keep_pca_btn"):
                            st.session_state.df = pca_df
                            st.success("PCA transformation saved!")
                            st.rerun()
                    else:
                        st.error("Could not perform PCA. Please check your data.")
                except Exception as e:
                    st.error(f"Error performing PCA: {str(e)}")

    elif processing_options == "Feature Selection":
        st.subheader("Feature Selection")
        
        # Check for numeric columns
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_columns) < 2:
            st.warning("Feature selection requires at least 2 numeric columns. Please convert or normalize your data first.")
            if st.button("Go to Column Conversion"):
                st.session_state.current_page = "Convert Columns"
                st.rerun()
        else:
            # Select target column
            target_column = st.selectbox("Select target column for feature selection:", df.columns.tolist())
            
            # Select number of features
            k = st.slider(
                "Select number of features to keep",
                min_value=1,
                max_value=len(numeric_columns),
                value=min(10, len(numeric_columns))
            )
            
            if st.button("Select Features", key="select_features_btn"):
                try:
                    # Perform feature selection
                    selected_df, selected_features = select_k_best_features(df, target_column, k)
                    
                    # Display results
                    st.success("Feature selection completed successfully!")
                    st.write("### Selected Features")
                    st.write(", ".join(selected_features))
                    
                    # Show transformed data
                    st.write("### Transformed Data")
                    st.dataframe(selected_df)
                    
                    # Option to save transformed data
                    if st.button("Keep Selected Features", key="keep_features_btn"):
                        st.session_state.df = selected_df
                        st.success("Selected features saved!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error performing feature selection: {str(e)}")

    else:  # View EDA
        # Add Column Type Analysis at the top
        st.write("### Column Type Analysis")
        column_types = identify_column_types(df)
        
        # Create DataFrame for better visualization
        type_df = pd.DataFrame(
            [(col, type_) for col, type_ in column_types.items()],
            columns=['Column', 'Type']
        )
        
        # Show table of column types
        st.dataframe(type_df)
        
        # Show distribution of column types
        st.write("### Distribution of Column Types")
        type_counts = pd.Series(column_types.values()).value_counts()
        fig, ax = plt.subplots(figsize=(10, 5))
        type_counts.plot(kind='bar')
        plt.title('Distribution of Column Types')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Group columns by type
        st.write("### Columns Grouped by Type")
        for type_ in sorted(set(column_types.values())):
            cols = [col for col, t in column_types.items() if t == type_]
            with st.expander(f"{type_.title()} Columns ({len(cols)})"):
                st.write(", ".join(cols))
        
        # Continue with existing EDA
        st.write("### Basic Dataset Information")
        st.write(f"Shape of dataset: {df.shape}")
        st.write("Column Names:", df.columns.tolist())

        # Missing Values
        st.write("### Missing Value Analysis")
        missing_values = df.isnull().sum()
        st.write(missing_values[missing_values > 0])

        # Data Types
        st.write("### Data Types")
        st.write(df.dtypes)

        # Basic Statistics
        st.write("### Basic Statistics")
        st.write(df.describe(include='all'))

        # Correlation Heatmap
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_columns) >= 2:
            st.write("### Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

        # Categorical Analysis
        st.write("### Categorical Columns Analysis")
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            st.write(f"#### Value Counts for {col}")
            st.bar_chart(df[col].value_counts())

        # String Pattern Analysis
        if len(categorical_columns) > 0:
            st.write("### String Pattern Analysis")
            pattern_analysis = {}
            for col in categorical_columns:
                if df[col].dtype == 'object':
                    lengths = df[col].str.len()
                    word_counts = df[col].str.split().str.len()
                    pattern_analysis[col] = {
                        'Average Length': lengths.mean(),
                        'Average Word Count': word_counts.mean()
                    }
            st.write(pd.DataFrame(pattern_analysis).T)
