# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set environment variables
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Set a theme for plots using Seaborn
sns.set_theme(style="whitegrid", palette="pastel")

# Initialize the LangChain model
chat_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Prompt template for the model
prompt_template = ChatPromptTemplate.from_template("You are an assistant knowledgeable in data preprocessing. Answer the following question based on the dataset: {question}")

# Streamlit App Title
st.title("Raw2Refined: Automated Data Preprocessing with LLM ")

# Upload CSV File
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)

    # Display dataset info
    st.write("### Dataset Overview")
    st.write(f"**Total rows:** {df.shape[0]}")
    st.write(f"**Total columns:** {df.shape[1]}")

    # Display the first 5 rows
    st.write("### First 5 Rows of the Dataset")
    st.write(df.head())

    # Display data types
    st.write("### Data Types of Columns")
    st.write(df.dtypes)

    # Show Count of numerical and categorical columns
    st.write("### Numerical columns and Categorical columns count")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    st.write(f"**Numerical columns count:** {len(numerical_columns)}")
    st.write(f"**Categorical columns count:** {len(categorical_columns)}")

    # Show missing values
    st.write("### Missing Values in the Dataset")
    missing_values = df.isnull().sum()
    st.write(missing_values[missing_values > 0])

    # Display a boxplot for numerical columns
    st.write("### Outliers Detection using Boxplot")
    numerical_columns = df.select_dtypes(include=["number"]).columns
    for col in numerical_columns:
        st.write(f"#### {col}")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=col, ax=ax)
        st.pyplot(fig)

        # Short summary of the boxplot
        # Calculate the Interquartile Range (IQR) to detect outliers
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Count of outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

        # Provide a summary for the user to understand the boxplot
        if outliers.empty:
            st.write(f"**Summary**: The column '{col}' has no significant outliers. All values fall within a normal range based on the Interquartile Range (IQR).")
        else:
            st.write(f"**Summary**: The column '{col}' contains potential outliers.")
            st.write(f"- **Lower Bound:** {lower_bound:.2f}")
            st.write(f"- **Upper Bound:** {upper_bound:.2f}")
            st.write(f"- **Number of potential outliers:** {outliers.shape[0]}")

            st.write(f"The values outside this range are considered potential outliers. Outliers can significantly affect the analysis and may require further investigation. "
                     "These values might represent anomalies or rare cases, but they could also be valid data points in certain scenarios.")

    # Display statistical summary
    st.write("### Statistical Summary of Numerical Columns")
    st.write(df.describe())



    # Step 8: User queries for insights via Gemini
    st.write("### Ask Questions to Gemini for Informative Insights")
    user_question = st.text_input("Enter your question about the dataset:")
      
    # Simplify the dataset for LLM processing (selecting only relevant columns)
    simplified_df = df.select_dtypes(include=['float64', 'int64']).head(10)
    dataset_json = simplified_df.to_json(orient='split')

    max_rows = 100  
    limited_df = df.head(max_rows)
    limited_dataset_json = limited_df.to_json(orient='split')

    if st.button("Get Answer"):
        if user_question:
            # Construct a simplified and specific prompt for LLM
            prompt = (
                f"You are provided with a sample dataset in JSON format. Here is a simplified version: {limited_dataset_json}. "
                f"Please answer the following question based on this data: {user_question}."
            )

            # Handle API call with proper error handling
            try:
                # Proper message formatting for the LLM
                messages = [
                    {"role": "system", "content": "You are an assistant knowledgeable in data analysis."},
                    {"role": "user", "content": prompt}
                ]

                # Generate response from the LLM
                response = chat_llm.invoke(prompt)

                # Access the content of the response
                formatted_response = response.content.replace('\n', '\n\n')

                # Display the LLM's answer
                st.write("### Gemini's Answer")
                st.write(formatted_response)
            except Exception as e:
                st.error(f"An error occurred while generating the response: {e}")
        else:
            st.warning("Please enter a question.")
else:
    st.info("Please upload a dataset to begin.")