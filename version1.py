# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Load environment variables from .env file
load_dotenv()

# Set environment variables
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# Set a theme for plots using Seaborn
sns.set_theme(style="whitegrid", palette="pastel")
chat_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# prompt template for the model
prompt_template = ChatPromptTemplate.from_template("You are an assistant knowledgeable in data analysis. Answer the following question based on the dataset: {question}")

# Title of the Streamlit app
st.title("raw2refined")

# Step 1: File uploader for user to upload CSV files
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    # Step 2: Load the dataset using Pandas
    df = pd.read_csv(uploaded_file)

    # Step 3: Data cleaning check
    st.write("### Checking for Missing or Inconsistent Data")

    # Checking for missing values
    if df.isnull().values.any():
        st.warning("Your dataset contains missing values. Please clean the dataset and re-upload.")
        st.write(df.isnull().sum())  # Show columns with missing values
    else:
        st.success("No missing values detected. Proceeding with analysis.")

        # Display a preview of the dataset
        st.write("### Dataset Preview")
        st.dataframe(df.head()) 

        # Display basic dataset information
        st.write("### Basic Information about the Dataset")
        st.write(f"Number of rows: {df.shape[0]}")
        st.write(f"Number of columns: {df.shape[1]}")
        st.write("### Dataset Summary")
        st.write(df.describe())  # Show statistical summary

        # Identify numerical and categorical columns
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = df.select_dtypes(include=['object']).columns

        # Step 4: Auto-Generated Dashboard with Key Visualizations
        st.write("### Auto-Generated Dashboard")

        # 4.1 Distribution plots for numerical columns
        if len(numeric_columns) > 0:
            st.write("#### Distribution of Numerical Columns")
            for i in range(0, len(numeric_columns), 2):  # Show 2 columns in one row
                cols = st.columns(2)  # Create 2 columns side-by-side
                for idx, column in enumerate(numeric_columns[i:i + 2]):
                    with cols[idx]:  # Place the plots in separate columns
                        plt.figure(figsize=(6, 4))  # Make the plot smaller
                        sns.histplot(df[column], kde=True, color='lightblue', bins=30)
                        plt.title(f'Distribution of {column}')
                        st.pyplot(plt)

        # 4.2 Bar plots for categorical columns
        if len(categorical_columns) > 0:
            st.write("#### Distribution of Categorical Columns")
            for i in range(0, len(categorical_columns), 2):
                cols = st.columns(2)
                for idx, column in enumerate(categorical_columns[i:i + 2]):
                    with cols[idx]:
                        top_categories = df[column].value_counts().nlargest(10)
                        filtered_df = df[df[column].isin(top_categories.index)]
                        plt.figure(figsize=(6, 4))
                        sns.countplot(x=column, data=filtered_df, palette="Set2", order=top_categories.index)
                        plt.title(f'Top 10 Categories in {column}')
                        plt.xticks(rotation=45)
                        st.pyplot(plt)

        # 4.3 Pie charts for categorical columns
        if len(categorical_columns) > 0:
            st.write("#### Pie Charts for Categorical Columns")
            for i in range(0, len(categorical_columns), 2):
                cols = st.columns(2)
                for idx, column in enumerate(categorical_columns[i:i + 2]):
                    with cols[idx]:
                        pie_data = df[column].value_counts().nlargest(5)  
                        plt.figure(figsize=(6, 6))
                        plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', colors=sns.color_palette("Set3"))
                        plt.title(f'Pie Chart of {column}')
                        st.pyplot(plt)

     
        # Step 5: Query-based Dataset Filtering with Dropdown
        st.write("### Filter the Dataset by Unique Values in a Column")

        # Dropdown to select a column for filtering
        selected_filter_column = st.selectbox("Select a column to filter by unique values", df.columns)

        # Display unique values for the selected column
        unique_values = df[selected_filter_column].unique()

        # Dropdown to select one of the unique values for filtering
        selected_value = st.selectbox(f"Select a unique value from the '{selected_filter_column}' column", unique_values)

        # Filter the dataset based on the selected value
        filtered_df = df[df[selected_filter_column] == selected_value]
        st.write(f"### Filtered Dataset where `{selected_filter_column} == {selected_value}`")
        st.dataframe(filtered_df)


        # Step 8: Send queries to Gemini API
        st.write("### Ask a Question about the Dataset")
        user_question = st.text_area("Enter your question here")

        if st.button("Submit Query"):
            # Prepare API call to Gemini
            response = requests.post(GOOGLE_API_KEY, json={"question": user_question}, headers={"Authorization": f"Bearer {GEMINI_API_KEY}"})

            if response.status_code == 200:
                answer = response.json().get("answer", "No answer found.")
                st.success("Response from Gemini:")
                st.write(answer)
            else:
                st.error("Failed to get a response from Gemini API.")

