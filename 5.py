import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit App Title
st.title("Raw2Refined: Automated Data Preprocessing ")

# Initialize session state variables
if 'step' not in st.session_state:
    st.session_state.step = 0

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

    # Create DataFrames for displaying the column names
    numerical_df = pd.DataFrame(numerical_columns, columns=["Numerical Columns"])
    categorical_df = pd.DataFrame(categorical_columns, columns=["Categorical Columns"])

    # Display the numerical and categorical column names in table format
    st.write("#### Numerical Columns")
    st.table(numerical_df)
    st.write("#### Categorical Columns")
    st.table(categorical_df)

    # Show missing values
    st.write("### Missing Values in the Dataset")
    missing_values = df.isnull().sum()
    st.write(missing_values[missing_values > 0])

    # Display statistical summary
    st.write("### Statistical Summary of Numerical Columns")
    st.write(df.describe())

    # Streamlit app title
    st.write("### Outlier Detection")

    # Identify numerical columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

    # Function to detect outliers using IQR method
    def count_outliers_iqr(data):
        outlier_counts = {}
        for col in numerical_columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
            outlier_counts[col] = len(outliers)
        return outlier_counts

    # Detect and count outliers in numerical columns
    numerical_outlier_counts = count_outliers_iqr(df)

    # Show the count of outliers for numerical columns
    st.write("### Outlier Count in Numerical Columns:")
    for col, count in numerical_outlier_counts.items():
        st.write(f"**{col}:** {count} outliers")

    # Optional: Visualize the outliers for numerical columns using box plots
    st.write("### Boxplot Visualization of Outliers:")
    for col in numerical_columns:
        st.write(f"**Boxplot for {col}:**")

        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f"Boxplot of {col} to Show Outliers")

        # Show plot in Streamlit
        st.pyplot(fig)

    # Handling the missing values
    st.write("### Handling the Missing Values")

    # Step 1: Show columns with missing values and allow user to select which to drop
    if st.session_state.step == 0:  # Step 1
        st.write("### Step 1: Drop Columns with Missing Values")
        missing_columns = df.columns[df.isnull().any()].tolist()

        if missing_columns:
            drop_columns = st.multiselect(
                "Select columns to drop (Columns with missing values)", 
                options=missing_columns,
                key='drop_columns'
            )

            # Drop selected columns
            if drop_columns:
                df = df.drop(columns=drop_columns)
                st.write(f"Dropped columns: {drop_columns}")
        
        # Proceed to Step 2
        if st.button("Proceed to Step 2"):
            st.session_state.step = 1  # Move to the next step

    # Step 2: Handle missing values in remaining numerical columns
    if st.session_state.step == 1:  # Step 2
        st.write("### Step 2: Fill Missing Values in Numerical Columns")

        # Filter out numerical columns that still have missing values
        numerical_columns_with_missing = df.select_dtypes(include=['float64', 'int64']).columns[df.select_dtypes(include=['float64', 'int64']).isnull().any()].tolist()

        # Initialize a counter for unique keys
        counter = 0

        while numerical_columns_with_missing:
            selected_column = st.selectbox(
                "Select a numerical column to handle missing values",
                options=numerical_columns_with_missing,
                key=f'select_column_{counter}'  # Use counter for unique key
            )

            # Option to fill with mean or median
            fill_method = st.radio(
                f"How would you like to fill missing values in {selected_column}?",
                options=["Mean", "Median"],
                key=f'radio_fill_method_{counter}'  # Use counter for unique key
            )

            # Fill missing values based on selected method
            if fill_method == "Mean":
                df[selected_column] = df[selected_column].fillna(df[selected_column].mean())
            elif fill_method == "Median":
                df[selected_column] = df[selected_column].fillna(df[selected_column].median())

            st.write(f"Filled missing values in **{selected_column}** using **{fill_method}**.")

            # Update the remaining numerical columns with missing values
            numerical_columns_with_missing = df.select_dtypes(include=['float64', 'int64']).columns[df.select_dtypes(include=['float64', 'int64']).isnull().any()].tolist()

            # Increment counter for the next iteration
            counter += 1 

        # Check if all missing values are handled
        if not df.isnull().any().any():
            st.write("All missing values in numerical columns have been handled!")
        
        # Proceed to Step 3
        if st.button("Proceed to Step 3"):
            st.session_state.step = 2  # Move to the next step

    # Step 3: Fill Missing Values in Categorical Columns
    if st.session_state.step == 2:  # Step 3
        st.write("### Step 3: Fill Missing Values in Categorical Columns")

        # Filter out categorical columns that still have missing values
        categorical_columns_with_missing = df.select_dtypes(include=['object']).columns[df.select_dtypes(include=['object']).isnull().any()].tolist()

        # Initialize a list to keep track of processed columns
        processed_columns = []

        while categorical_columns_with_missing:
            # Allow users to select one categorical column to handle missing values
            selected_column = st.selectbox(
                "Select a categorical column to handle missing values:",
                options=categorical_columns_with_missing,
                key=f'select_categorical_column_{len(processed_columns)}'  # Unique key for each iteration
            )

            # Option to fill with mode or retain original
            fill_option = st.radio(
                f"How would you like to handle missing values in **{selected_column}**?",
                options=["Fill with Mode", "Leave as it is"],
                key=f'radio_fill_option_{len(processed_columns)}'  # Unique key for each iteration
            )

            # Handle the selection
            if fill_option == "Fill with Mode":
                df[selected_column] = df[selected_column].fillna(df[selected_column].mode()[0])
                st.write(f"Filled missing values in **{selected_column}** using **Mode**.")
            else:
                st.write(f"Retained missing values in **{selected_column}** as it is.")

            # Add the processed column to the list and update remaining columns
            processed_columns.append(selected_column)
            categorical_columns_with_missing = [col for col in categorical_columns_with_missing if col not in processed_columns]

        # Final check on categorical columns
        if not df.isnull().any().any():
            st.write("All missing values in categorical columns have been handled!")
            st.write("### ALL the Missing values handled successfully!!!")
            st.write(df)        
  
