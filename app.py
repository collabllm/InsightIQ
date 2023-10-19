import streamlit as st
import pandas as pd
import codecs
import sweetviz as sv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from PIL import Image
import psycopg2
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, classification_report
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier

from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from pandasai.callbacks import StdoutCallback
from langchain.agents import AgentType
from langchain.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI

import streamlit.components.v1 as stc
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

icon_url = "https://www2.deloitte.com/content/dam/Deloitte/in/Images/promo_images/in-deloitte-logo-1x1-noexp.png"
st.set_page_config(
    page_title="InsightIQ",
    page_icon=icon_url,
)

def st_display_sweetviz(report_html, width = 1000, height = 500):
    report_file = codecs.open(report_html, 'r')
    page = report_file.read()
    stc.html(page, width = width, height = height, scrolling = True)

def load_and_split_data(data):
    if isinstance(data, str):  # Check if data is a string (file path)
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):  # Check if data is a DataFrame
        df = data

    target_var = st.selectbox("Select the Target Variable: ", df.columns)

    task = st.selectbox("Select the Operation: ", ["Regression", "Classification"])

    missing_value_method = st.selectbox(
    "Select Missing Value Handling Method:",
    ["None (Do not handle Missing Values)", "Mean", "Median", "Mode"],
    )

    if missing_value_method != 'None (Do not handle Missing Values)':
        if missing_value_method == "Mean":
            df = df.fillna(df.mean())
        elif missing_value_method == "Median":
            df = df.fillna(df.median())
        elif missing_value_method == "Mode":
            df = df.fillna(df.mode().iloc[0])

    columns_to_encode = st.multiselect("Select columns to one-hot encode:", df.columns)

    if columns_to_encode:
        df = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)

    X = df.drop(columns=[target_var])
    y = df[target_var]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    k_value = st.slider("Select the value of K for K-Fold Cross-Validation:", 2, 10, 5)


    if st.button("Start Process"):
        model_results = []
        if task == "Regression":
            models = [LinearRegression(),
                      RandomForestRegressor(),
                      GradientBoostingRegressor(),
                      KNeighborsRegressor(),
                      MLPRegressor(),
                      SVR(),
                      # XGBRegressor(),
                      DecisionTreeRegressor()]

        elif task == "Classification":
            models = [LogisticRegression(),
                      RandomForestClassifier(),
                      GradientBoostingClassifier(),
                      KNeighborsClassifier(),
                      MLPClassifier(),
                      SVC(),
                      # XGBClassifier(),
                      DecisionTreeClassifier()]

        for model in models:
            with st.spinner(f"Training and evaluating {model.__class__.__name__}..."):
                model.fit(X_train, y_train)

                if task == "Regression":
                    predictions = model.predict(X_val)
                    r_squared = r2_score(y_val, predictions)
                    rmse = mean_squared_error(y_val, predictions, squared=False)
                    mae = mean_absolute_error(y_val, predictions)

                    st.subheader(f"Model: {model.__class__.__name__}")
                    st.write("R-squared (Coefficient of Determination):", r_squared)
                    st.write("Root Mean Squared Error:", rmse)
                    st.write("Mean Absolute Error:", mae)

                    if isinstance(model, RandomForestRegressor) or isinstance(model, GradientBoostingRegressor):
                        feature_importance = model.feature_importances_
                        feature_names = X.columns

                        # Create a DataFrame to store feature importance values
                        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
                        importance_df = importance_df.sort_values(by='Importance', ascending=False)

                        # Create a Plotly bar chart for feature importance
                        fig = px.bar(importance_df, x='Feature', y='Importance', title='Feature Importance')
                        st.plotly_chart(fig, use_container_width=True)

                elif task == "Classification":
                    predictions = model.predict(X_val)
                    accuracy = accuracy_score(y_val, predictions)
                    classification_rep = classification_report(y_val, predictions)

                    st.subheader(f"Model: {model.__class__.__name__}")
                    st.write("Accuracy:", accuracy)
                    st.write("Classification Report:")
                    st.text(classification_rep)

                    # Feature Importance for Classification
                    if isinstance(model, RandomForestClassifier) or isinstance(model, GradientBoostingClassifier):
                        feature_importance = model.feature_importances_
                        feature_names = X.columns

                        # Create a DataFrame to store feature importance values
                        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
                        importance_df = importance_df.sort_values(by='Importance', ascending=False)

                        # Create a Plotly bar chart for feature importance
                        fig = px.bar(importance_df, x='Feature', y='Importance', title='Feature Importance')
                        st.plotly_chart(fig, use_container_width=True)

                if task == "Regression":
                    model_results.append((model.__class__.__name__, r_squared, rmse, mae))
                elif task == "Classification":
                    model_results.append((model.__class__.__name__, accuracy))

                # K-Fold Cross Validation
                st.write("**K-Fold Cross Validation Scores**")
                cv = KFold(n_splits = k_value, shuffle = True, random_state = 42)
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv)
                st.write("Mean Score: ", cv_scores.mean())
                st.write("Standard Deviation: ", cv_scores.std())

                st.write('---')

        model_results.sort(key=lambda x: x[1], reverse=True)

        st.subheader("Model Comparison")

        # Display the performance data in a tabular format
        if task == "Regression":
            column_headers = ["Model Name", "R-squared", "RMSE", "MAE"]
            df_model_results = pd.DataFrame(model_results, columns = column_headers)
            st.table(df_model_results)

        elif task == "Classification":
            column_headers = ["Model Name", "Accuracy"]
            df_model_results = pd.DataFrame(model_results, columns = column_headers)
            st.table(df_model_results)


def fetch_data_from_database(database_url, table_name):
    try:
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        query = f"SELECT * FROM {table_name};"
        cursor.execute(query)
        df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
        return df
    except Exception as e:
        print(f"Error: {str(e)}")
        return None
    finally:
        if conn:
            cursor.close()
            conn.close()

st.markdown("""
<style>
    .stButton button {
        border: 2px solid #007e3a;
    }

    .stButton button:hover {
        background-color:#007e3a;
        border: 2px solid #007e3a;
        color: white;
    }
    
    .stSlider .stSliderContext .streamlit-progress-bar {{
        background: #007e3a;
    }}

    .stSlider .stSliderContext .streamlit-slider {{
        background: #007e3a;
    }}

    .stSlider .stSliderContext .streamlit-thumb {{
        background: #007e3a;
        border: 2px solid #007e3a;
    }}

    .stSlider .stSliderContext .streamlit-thumb:hover {{
        background: #fff;
        border: 2px solid #007e3a;
    }}

</style>
""", unsafe_allow_html=True)

def main():

    file_uploaded = None
    df = None

    menu = ["Home", "AutoProfiler", "AutoViz", "AutoML", "ProfilerReport", "DataSpeak"]
    choice = st.sidebar.selectbox("Menu", menu)

    data_source = st.sidebar.selectbox("Select Data Source", ["CSV File", "Database"])

    if data_source == "CSV File":
        with st.sidebar.header('Upload your CSV Data'):
            file_uploaded = st.sidebar.file_uploader("Input your CSV File", type=["csv"])

        if file_uploaded is not None:
            @st.cache_data
            def load_csv():
                csv = pd.read_csv(file_uploaded)
                return csv
                    
            df = load_csv()
        
    elif data_source == "Database":

        st.sidebar.subheader("Database Connection")
        db_username = st.sidebar.text_input("Database Username", "postgres")
        db_password = st.sidebar.text_input("Database Password", "", type="password")
        db_host = st.sidebar.text_input("Database Host", "localhost")
        db_port = st.sidebar.number_input("Database Port", 5432)
        db_name = st.sidebar.text_input("Schema Name", "dvdrental")
    
        if db_password:
            database_url = f"postgresql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
            
            action = st.sidebar.selectbox("Choose an action", ["List Tables", "Custom Query"])
            
            if action == "List Tables":
                if not database_url:
                    st.warning("Please enter a valid database URL.")
                else:
                    # Fetch table names from the database
                    conn = psycopg2.connect(database_url)
                    cursor = conn.cursor()
                    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
                    tables = cursor.fetchall()
                    conn.close()

                    # Extract the table names as a list of strings
                    table_names = [table[0] for table in tables]

                    if table_names:
                        # Create a choice for selecting a table
                        selected_table = st.sidebar.selectbox("Select Table", table_names)

                        if st.sidebar.button("Fetch Data"):
                            # Fetch data from the selected table
                            df = fetch_data_from_database(database_url, selected_table)
            
            elif action == "Custom Query":
                custom_query = st.sidebar.text_area("Input Custom Query", "")
                if st.sidebar.button("Execute Query"):
                    if not database_url:
                        st.warning("Please enter a valid database URL.")
                    elif not custom_query:
                        st.warning("Please enter a custom SQL query.")
                    else:
                        try:
                            conn = psycopg2.connect(database_url)
                            cursor = conn.cursor()
                            cursor.execute(custom_query)
                            result = cursor.fetchall()

                            column_names = [desc[0] for desc in cursor.description]

                            conn.close()
                            
                            if result:
                                df = pd.DataFrame(result, columns = column_names)
                        except Exception as e:
                            st.error(f"Error executing the custom query: {str(e)}")

    if "session_df" not in st.session_state:
        st.session_state.session_df = None

    if file_uploaded is not None or (data_source == "Database" and df is not None):
        # Set df to session_state to retain it
        st.session_state.session_df = df

    if choice == "Home":
        image_url = "https://www2.deloitte.com/content/dam/assets/logos/deloitte.svg"
        st.image(image_url, width=100)
        st.title("InsightIQ")
        st.write("This Streamlit application provides various tools for data profiling, visualization, automated machine learning, and more.")

        st.subheader("Our Offerings")
        st.markdown("- **AutoProfiler**: Get insights into your dataset, including statistics, data types, and missing values.")
        st.markdown("- **AutoViz**: Visualize your dataset with various plots and charts.")
        st.markdown("- **AutoML**: Train and evaluate machine learning models on your data.")
        st.markdown("- **ProfilerReport**: Generate detailed data profiling reports.")
        st.markdown("- **DataSpeak**: Work in Progress  ")

        st.subheader("Instructions")
        st.markdown("1. Upload your CSV data file in the sidebar or connect your database to select a table of your choice.")
        st.markdown("2. Choose a menu option from the sidebar to use the various tools we have to offer.")

    if choice == "AutoProfiler":
        st.header("AutoProfiler")
        st.caption("This streamlit web application performs automated data profiling and analysis for an uploaded CSV file. You can upload a dataset, and the application provides various insights and alerts about the data.")
        
        df = st.session_state.session_df

        if df is not None:

            corr_matrix = df.corr()

            # Create a button to show/hide alerts
            show_alerts = st.button("Show Alerts")
            alerts_visible = False

            if show_alerts:
                alerts_visible = not alerts_visible

            if alerts_visible:
                # Initialize a list to store all alerts
                all_alerts = []

                # Correlation Alerts
                correlation_alerts = []
                for column1 in corr_matrix.columns:
                    for column2 in corr_matrix.index:
                        if column1 != column2 and abs(corr_matrix.loc[column1, column2]) >= 0.8:
                            alert = f"- {column1} is Highly overall Correlated with {column2} - High correlation"
                            correlation_alerts.append(alert)

                if correlation_alerts:
                    all_alerts.extend(correlation_alerts)

                # Object Data Type Alerts
                object_type_columns = df.select_dtypes(include=['object']).columns
                object_alerts = [f"- {column} is of type 'object'" for column in object_type_columns]

                if object_alerts:
                    all_alerts.extend(object_alerts)

                # Imbalance Alerts
                imbalance_threshold = 0.5
                imbalance_alerts = []
                for column in df.columns:
                    if df[column].value_counts(normalize=True).max() >= imbalance_threshold:
                        imbalance_percentage = df[column].value_counts(normalize=True).max() * 100
                        alert = f"- {column} is Highly Imbalanced ({imbalance_percentage:.1f}%)"
                        imbalance_alerts.append(alert)

                if imbalance_alerts:
                    all_alerts.extend(imbalance_alerts)

                # Date Format Alerts
                date_alerts = []
                date_keyword = "date"
                for column in df.columns:
                    if isinstance(column, str) and date_keyword in column.lower():
                        try:
                            pd.to_datetime(df[column], errors='raise')
                        except ValueError:
                            alert = f"- {column} contains '{date_keyword}' but is not in a valid date-time format"
                            date_alerts.append(alert)

                if date_alerts:
                    all_alerts.extend(date_alerts)

                # Zero Alerts
                zero_threshold = 0.8
                zero_alerts = []
                for column in df.columns:
                    zero_percentage = (df[column] == 0).sum() / len(df)
                    if zero_percentage >= zero_threshold:
                        alert = f"- {column} has {zero_percentage:.1f}% Zeroes"
                        zero_alerts.append(alert)

                if zero_alerts:
                    all_alerts.extend(zero_alerts)

                # Display the alerts as bullet points in a Markdown list
                if all_alerts:
                    st.subheader("Alerts")
                    for alert in all_alerts:
                        st.markdown(alert)

            # Add a button to hide the alerts
            if alerts_visible:
                hide_alerts = st.button("Hide Alerts")
                if hide_alerts:
                    alerts_visible = False


            
            st.subheader("Input Data")
            st.write(df.head())

            st.write("---")
            
            st.subheader("Shape of the Dataset")
            st.write(df.shape)
            
            st.write("---")

            st.subheader("Data Types of the Features")
            st.write(df.dtypes)

            st.write("---")

            missing_count = df.isnull().sum()
            st.subheader("Missing Value Count for Each Column")
            st.table(missing_count)

            st.write("---")

            unique_counts = df.nunique()
            st.subheader("Number of Unique Values in Each Column")
            st.table(unique_counts)

            st.write("---")

            correlations = df.corr()
            st.subheader("Correlation Table")
            st.table(correlations)

            st.write("---")

            corr_matrix = df.corr()

            st.subheader('Correlation Heatmap')

            fig = px.imshow(corr_matrix, color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)

        else:
            df = None
            st.info("Waiting for the Data File to be uploaded.")


    if choice == "AutoViz":
        st.header("AutoViz")
        st.write("AutoViz is a powerful automated data analysis and visualization tool designed to simplify and streamline the process of exploring and gaining insights from your data.")

        df = st.session_state.session_df

        if df is not None:

            st.subheader("Uploaded Dataset")
            st.write(df.head())

            st.write("---")

            # Basic dataset information
            st.subheader("Dataset Info")
            st.write(f"Number of rows: {df.shape[0]}")
            st.write(f"Number of columns: {df.shape[1]}")

            st.write("---")

            # Data type distribution
            st.subheader("Data Type Distribution of Columns")
            st.write(df.dtypes.value_counts())

            st.write("---")

            # Missing values
            missing_percentage = df.isnull().mean() * 100

            st.subheader("Missing Value Percentage")
            fig = px.bar(missing_percentage, x=missing_percentage.index, y=missing_percentage.values,
                labels={'x': 'Feature', 'y': 'Percentage'}, title='Missing Value Percentage',
                color=missing_percentage.values, color_continuous_scale='Viridis')

            # Update layout for better appearance
            fig.update_layout(xaxis_tickangle=-45)

            st.plotly_chart(fig, use_container_width=True)


            st.write("---")
            # Correlation Heatmap
            corr_matrix = df.corr()

            st.subheader('Correlation Heatmap')

            fig = px.imshow(corr_matrix, color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)

            st.write("---")

            # Numeric columns statistics
            st.subheader("Numeric Columns Statistics")
            st.write(df.describe())

            st.write("---")

            # Categorical columns distribution
            st.subheader("Categorical Columns Distribution")
            categorical_columns = df.select_dtypes(include=['object']).columns
            for column in categorical_columns:
                st.write(f"**{column}**")
                st.write(df[column].value_counts())

        else:
            st.info("Waiting for the Data File to be uploaded.")

    if choice == "AutoML":
        st.header("AutoML")
        st.write("**Effortlessly Build Powerful Models**") 
        st.write("Our AutoML tool simplifies machine learning by automatically training models on your data, generating results, and highlighting feature importance. Just upload your dataset and let AutoML do the rest, delivering accurate insights in no time.")

        df = st.session_state.session_df

        if df is not None:
            load_and_split_data(df)
        else:
            st.info("Waiting for the Data File to be uploaded.")

    if choice == "DataSpeak":
        st.header("DataSpeak")
        st.write("**Unlock the Power of Conversational Data Analysis with DataSpeak**") 
        st.write("**DataSpeak** let's you seamlessly explore and interact with your data, ask questions in natural language, and receive instant insights and recommendations.")

        df = st.session_state.session_df

        if df is not None:
            openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
            if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
                st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])

            if prompt := st.chat_input(placeholder="What is the data about?"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)

                if not openai_api_key:
                    st.info("Please add your OpenAI API key to continue.")
                    st.stop()

                llm = ChatOpenAI(
                    temperature=0, model="gpt-3.5-turbo-0613", openai_api_key=openai_api_key, streaming=True
                )

                pandas_df_agent = create_pandas_dataframe_agent(
                    llm,
                    df,
                    verbose=True,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                    handle_parsing_errors=True,
                )

                with st.chat_message("assistant"):
                    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                    response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.write(response)

        else:
            st.info("Waiting for the Data File to be uploaded.")

    if choice == "ProfilerReport":
        st.header("Automated Profiling Report of Data using Profiler Report")
        st.write("Effortlessly unveil key insights and patterns within your data using Profiler Report tool which provides a comprehensive summary, revealing data types, missing values, correlations, and more, simplifying data exploration and analysis.")

        df = st.session_state.session_df

        if df is not None:
            if "data_report" not in st.session_state:
                profile = ProfileReport(df, explorative=True)
                st.session_state.data_report = profile
            else:
                profile = st.session_state.data_report
            
            st.header("**Input Data**")
            st.write(df.head())

            st.write("---")
            
            st.subheader("Data Profiling Report")
            st_profile_report(profile)
            
        else:
            st.info("Waiting for the CSV File to be uploaded.")

if __name__ == '__main__':
    main()