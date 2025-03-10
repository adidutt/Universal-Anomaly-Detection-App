import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from pyod.models.hbos import HBOS
from loguru import logger
import plotly.express as px
import os

# Configure logging with loguru
logger.remove()  # Remove default handler
logger.add("anomaly_detection.log", rotation="500 MB", level="INFO")
logger.add(lambda msg: st.sidebar.text(msg), level="INFO")  # Display logs in Streamlit sidebar

class UniversalAnomalyDetector:
    def __init__(self, contamination='auto', n_models=3, random_state=42):
        """
        Initialize the UniversalAnomalyDetector.

        Parameters:
        - contamination: 'auto' or float between 0 and 0.5 (fraction of anomalies)
        - n_models: Number of models to use in the ensemble (1-4)
        - random_state: Seed for reproducibility
        """
        self.contamination = contamination
        self.n_models = n_models
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.models = []
        self.feature_names = None
        self.df_preprocessed = None

    def preprocess(self, df):
        """Preprocess the input DataFrame: impute missing values, encode categoricals, and scale."""
        logger.info("Starting preprocessing of data")
        try:
            # Separate numerical and categorical columns
            df_numeric = df.select_dtypes(include=[np.number])
            df_categorical = df.select_dtypes(exclude=[np.number])

            # Handle categorical data
            if not df_categorical.empty:
                logger.info(f"Encoding categorical columns: {list(df_categorical.columns)}")
                encoded = self.encoder.fit_transform(df_categorical)
                encoded_df = pd.DataFrame(encoded, index=df.index,
                                        columns=self.encoder.get_feature_names_out())
                df_preprocessed = pd.concat([df_numeric, encoded_df], axis=1)
            else:
                df_preprocessed = df_numeric.copy()

            # Store feature names for later use
            self.feature_names = df_preprocessed.columns

            # Impute missing values and scale
            df_preprocessed = pd.DataFrame(self.imputer.fit_transform(df_preprocessed),
                                        index=df_preprocessed.index,
                                        columns=self.feature_names)
            df_preprocessed = pd.DataFrame(self.scaler.fit_transform(df_preprocessed),
                                        index=df_preprocessed.index,
                                        columns=self.feature_names)
            self.df_preprocessed = df_preprocessed
            logger.info("Preprocessing completed")
            return df_preprocessed
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def estimate_contamination(self, df_preprocessed):
        """Estimate contamination using the IQR method."""
        logger.info("Estimating contamination using IQR method")
        try:
            outliers = set()
            for col in df_preprocessed.columns:
                Q1 = df_preprocessed[col].quantile(0.25)
                Q3 = df_preprocessed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                col_outliers = df_preprocessed[(df_preprocessed[col] < lower_bound) |
                                            (df_preprocessed[col] > upper_bound)].index
                outliers.update(col_outliers)
            contamination = len(outliers) / len(df_preprocessed) if outliers else 0.01
            contamination = max(0.01, min(contamination, 0.5))  # Bound between 0.01 and 0.5
            logger.info(f"Estimated contamination: {contamination}")
            return contamination
        except Exception as e:
            logger.error(f"Error in contamination estimation: {str(e)}")
            raise

    def select_models(self, df_preprocessed):
        """Select a diverse set of anomaly detection models."""
        logger.info(f"Selecting {self.n_models} models for anomaly detection")
        try:
            models = [
                IsolationForest(contamination=self.contamination if self.contamination != 'auto' else 0.1,
                                random_state=self.random_state),
                LocalOutlierFactor(n_neighbors=min(20, df_preprocessed.shape[0]-1),
                                contamination=self.contamination if self.contamination != 'auto' else 0.1),
                GaussianMixture(n_components=2, random_state=self.random_state),
                HBOS()
            ]
            return models[:self.n_models]
        except Exception as e:
            logger.error(f"Error in model selection: {str(e)}")
            raise

    def fit_predict(self, df):
        """Fit the ensemble of models and predict anomalies."""
        logger.info("Fitting models and predicting anomalies")
        try:
            df_preprocessed = self.preprocess(df)
            if self.contamination == 'auto':
                self.contamination = self.estimate_contamination(df_preprocessed)
            self.models = self.select_models(df_preprocessed)

            scores = []
            for model in self.models:
                if isinstance(model, IsolationForest):
                    model.fit(df_preprocessed)
                    score = model.decision_function(df_preprocessed)
                    score = (score < np.percentile(score, 100 * self.contamination)).astype(int)
                elif isinstance(model, LocalOutlierFactor):
                    pred = model.fit_predict(df_preprocessed)
                    score = (pred == -1).astype(int)
                elif isinstance(model, GaussianMixture):
                    model.fit(df_preprocessed)
                    score = model.score_samples(df_preprocessed)
                    score = (score < np.percentile(score, 100 * self.contamination)).astype(int)
                elif isinstance(model, HBOS):
                    model.fit(df_preprocessed)
                    score = model.decision_function(df_preprocessed)
                    score = (score > np.percentile(score, 100 * (1 - self.contamination))).astype(int)
                scores.append(score)

            # Ensemble scoring via majority voting
            ensemble_scores = np.array(scores).mean(axis=0)
            anomalies = (ensemble_scores > 0.5).astype(bool)
            logger.info("Anomaly detection completed")
            return anomalies, ensemble_scores
        except Exception as e:
            logger.error(f"Error in fit_predict: {str(e)}")
            raise

    def explain_anomalies(self, df, anomalies):
        """Generate explanations for why each row was flagged as an anomaly."""
        logger.info("Generating explanations for anomalies")
        try:
            explanations = []
            for idx, is_anomaly in enumerate(anomalies):
                if is_anomaly:
                    row = self.df_preprocessed.iloc[idx]
                    # Identify features with extreme values (beyond 2 standard deviations)
                    extreme_features = [col for col in self.feature_names if abs(row[col]) > 2]
                    if extreme_features:
                        explanation = f"Extreme values in: {', '.join(extreme_features)}"
                    else:
                        explanation = "Detected by ensemble voting"
                    explanations.append(explanation)
                else:
                    explanations.append("Not an anomaly")
            return explanations
        except Exception as e:
            logger.error(f"Error in explain_anomalies: {str(e)}")
            raise

    def detect_anomalies(self, df):
        """Detect anomalies and return a DataFrame with results and explanations."""
        try:
            anomalies, scores = self.fit_predict(df)
            explanations = self.explain_anomalies(df, anomalies)
            result_df = df.copy()
            result_df['Anomaly'] = anomalies
            result_df['Anomaly_Score'] = scores
            result_df['Explanation'] = explanations
            # Sort the DataFrame so anomalies are at the top
            result_df = result_df.sort_values(by='Anomaly', ascending=False)
            return result_df
        except Exception as e:
            logger.error(f"Error in detect_anomalies: {str(e)}")
            raise

def detect_datetime_columns(df):
    """Automatically detect potential datetime columns."""
    datetime_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        elif df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col])
                datetime_cols.append(col)
            except:
                pass
    return datetime_cols

def load_data(file):
    """Load data from a CSV or Excel file."""
    logger.info(f"Loading file: {file.name}")
    try:
        file_ext = os.path.splitext(file.name)[1].lower()
        if file_ext == '.csv':
            return pd.read_csv(file)
        elif file_ext in ['.xlsx', '.xls']:
            return pd.read_excel(file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        st.error(f"Error loading file: {str(e)}")
        return None

def main():
    """Streamlit app for anomaly detection."""
    st.title("Universal Anomaly Detection App")
    st.write("Upload your data (CSV or Excel) to detect anomalies and understand why they were flagged.")

    # Sidebar for logs
    st.sidebar.title("Activity Log")

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.write("### Input Data Preview")
            st.dataframe(df.head())

            # Automatically detect datetime columns
            datetime_cols = detect_datetime_columns(df)
            if datetime_cols:
                st.write(f"Detected potential datetime columns: {', '.join(datetime_cols)}")
                time_col = st.selectbox("Select Time Column", options=['None'] + datetime_cols)
            else:
                time_col = st.selectbox("Select Time Column (if any)", options=['None'] + list(df.columns))

            # Visualize input data
            st.write("### Visualize Input Data")
            if time_col != 'None':
                try:
                    df[time_col] = pd.to_datetime(df[time_col])
                    value_col = st.selectbox("Select Value Column for Time Series",
                                           options=[col for col in df.columns if col != time_col])
                    if value_col:
                        fig = px.line(df, x=time_col, y=value_col, title=f"Time Series: {value_col} over {time_col}")
                        st.plotly_chart(fig, use_container_width=True)
                except:
                    st.error("Could not convert selected column to datetime.")

            # Anomaly detection parameters
            st.write("### Anomaly Detection Parameters")
            contamination_option = st.radio("Contamination Setting", ["Set manually", "Estimate automatically"])
            if contamination_option == "Set manually":
                contamination = st.slider("Contamination", 0.01, 0.5, 0.01)
            else:
                contamination = 'auto'
            n_models = st.slider("Number of Models", 1, 4, 3)

            if st.button("Detect Anomalies"):
                detector = UniversalAnomalyDetector(contamination=contamination, n_models=n_models)
                result_df = detector.detect_anomalies(df)

                # Display results with horizontal and vertical scrolling
                st.write("### Results: DataFrame with Anomalies")
                st.dataframe(result_df, height=400, width=1000)

                # Filter and show anomalies
                anomalies_df = result_df[result_df['Anomaly']]
                st.write(f"### Detected Anomalies ({len(anomalies_df)} found)")
                st.dataframe(anomalies_df, height=400, width=1000)

                # Visualization of anomalies
                st.write("### Anomaly Visualization")
                if time_col != 'None' and value_col:
                    fig = px.scatter(result_df, x=time_col, y=value_col, color='Anomaly',
                                   title="Time Series with Anomalies",
                                   color_discrete_map={True: 'red', False: 'blue'},
                                   hover_data=['Anomaly_Score', 'Explanation'])
                    st.plotly_chart(fig, use_container_width=True)
                elif len(df.columns) >= 2:
                    x_col = st.selectbox("X-axis", options=df.columns)
                    y_col = st.selectbox("Y-axis", options=[col for col in df.columns if col != x_col])
                    fig = px.scatter(result_df, x=x_col, y=y_col, color='Anomaly',
                                   title="Scatter Plot of Anomalies",
                                   color_discrete_map={True: 'red', False: 'blue'},
                                   hover_data=['Anomaly_Score', 'Explanation'])
                    st.plotly_chart(fig, use_container_width=True)

                # Download results
                csv = result_df.to_csv(index=False)
                st.download_button("Download Results", csv, "anomalies_detected.csv", "text/csv")

if __name__ == "__main__":
    main()