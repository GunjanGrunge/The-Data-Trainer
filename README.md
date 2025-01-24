# Column Mapping Analysis Application

## Setup Instructions

### 1. Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### 2. Install Dependencies
Open a terminal/command prompt and run the following commands:

```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install streamlit pandas numpy scikit-learn tensorflow xgboost seaborn matplotlib openpyxl xlrd scipy
```

### 3. Running the Application

1. Navigate to the application directory:
```bash
cd path/to/ColumnMapping
```

2. Run the Streamlit application:
```bash
streamlit run app.py
```

The application will automatically open in your default web browser. If it doesn't, you can access it at:
- Local URL: http://localhost:8501
- Network URL: http://your-ip-address:8501

### 4. Using the Application

1. **Home Page**:
   - Upload your Excel file (.xlsx, .xls, .xlsm)
   - Preview the data
   - Click "Proceed to Analysis"

2. **Analysis Features**:
   - Data Cleaning
   - Missing Value Handling
   - Data Normalization
   - Column Type Conversion
   - PCA Analysis
   - Feature Selection
   - Model Training (Multiple algorithms available)
   - Model Testing

3. **Model Training Options**:
   - Decision Tree
   - Random Forest
   - SVM
   - XGBoost
   - Neural Network
   - Ensemble Methods

### 5. Troubleshooting

If you encounter any errors:

1. Ensure all dependencies are installed:
```bash
pip install -r requirements.txt  # if available
```

2. Update Streamlit:
```bash
pip install --upgrade streamlit
```

3. Common issues:
   - Memory errors: Reduce dataset size or increase system memory
   - Module not found: Run `pip install [module_name]`
   - Excel read errors: Install `openpyxl` and `xlrd`

### 6. System Requirements

- Minimum 4GB RAM (8GB recommended)
- 2GB free disk space
- Modern web browser (Chrome/Firefox recommended)

### 7. Support

For issues and questions:
1. Check error messages in terminal
2. Verify data format matches requirements
3. Contact technical support or raise an issue on repository

### 8. Additional Tips

- Keep Excel files under 100MB for optimal performance
- Clean data before uploading for better results
- Save trained models for future use
- Use the comparison feature to evaluate different models