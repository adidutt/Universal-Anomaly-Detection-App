# Universal Anomaly Detection App

## Overview
The **Universal Anomaly Detection App** is a Streamlit-based web application that allows users to upload datasets (CSV or Excel) and automatically detect anomalies in their data. The application leverages multiple anomaly detection models, including:

- **Isolation Forest**
- **Local Outlier Factor (LOF)**
- **Gaussian Mixture Model (GMM)**
- **Histogram-Based Outlier Score (HBOS)**

The app preprocesses the dataset, estimates contamination levels, selects anomaly detection models, and provides visualizations to help users interpret the results.

## Features
- **Automated Data Preprocessing**: Handles missing values, categorical encoding, and feature scaling.
- **Anomaly Detection Models**: Uses an ensemble of up to four different models for robust anomaly detection.
- **Interactive Visualizations**: Provides time-series and scatter plots using Plotly to display anomalies.
- **User-Friendly UI**: Built with Streamlit for easy interaction.
- **Logging**: Logs events and errors using Loguru, with logs displayed in the Streamlit sidebar.
- **Downloadable Results**: Allows users to download the processed dataset with anomaly labels and explanations.

## Installation
To set up this project on your local machine, follow these steps:

### Prerequisites
Ensure you have Python 3.7+ installed. You also need `pip` to install dependencies.

### Steps
1. **Clone the repository**:
   ```sh
   git clone https://github.com/yourusername/universal-anomaly-detection.git
   cd universal-anomaly-detection
   ```
2. **Create a virtual environment (optional but recommended)**:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```
4. **Run the application**:
   ```sh
   streamlit run app.py
   ```

## Usage
1. Open the Streamlit UI in your browser (it should automatically launch after running the above command).
2. Upload a CSV or Excel file containing your dataset.
3. Configure anomaly detection settings (contamination level, model selection, etc.).
4. View detected anomalies in tabular form and through interactive visualizations.
5. Download the processed results with anomaly labels and explanations.

## File Structure
```
/your_project_directory
│── anomaly_detection.py   # Core anomaly detection logic
│── app.py                 # Streamlit app
│── requirements.txt       # List of dependencies
│── README.md              # Documentation
│── anomaly_detection.log  # Log file (generated at runtime)
```

## Technologies Used
- **Python** (Data processing and modeling)
- **Streamlit** (Web application framework)
- **Scikit-Learn** (Machine learning models)
- **PyOD** (Outlier detection library)
- **Pandas & NumPy** (Data manipulation and processing)
- **Plotly** (Data visualization)
- **Loguru** (Logging management)

## Contribution
If you would like to contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a Pull Request.
