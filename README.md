# Raw2Refined: Automated Data Preprocessing

**Raw2Refined** is an interactive and intuitive web application built with Streamlit that automates the entire data preprocessing workflow. It enables users to upload CSV files, handle missing data, detect and manage outliers, remove duplicates, and gain insights from both numerical and categorical columns. This tool provides a seamless interface for users looking to clean and prepare their datasets for further analysis or modeling.

---

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [How to Use](#how-to-use)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **CSV Upload:** Upload any CSV file and view the first 5 rows of the dataset.
- **Data Overview:** Displays the total number of rows, columns, and data types.
- **Missing Value Detection:** Identifies columns with missing values and provides options to handle them.
- **Outlier Detection:** Automatically detects outliers using the IQR method and offers to remove them.
- **Handling Duplicates:** Detects and removes duplicate rows.
- **Interactive Data Preprocessing:** Users can choose methods for filling missing values (mean, median, mode), drop columns, and more.
- **Processed Data Overview:** Displays the final dataset after preprocessing is complete.

---

## Getting Started

### Prerequisites

Before running the app, ensure you have Python 3.x installed on your machine. You will also need the following libraries:

- `pandas`
- `numpy`
- `streamlit`
- `matplotlib`
- `seaborn`
- `scikit-learn`

### Installation

1. **Clone this repository**:

    ```bash
    git clone https://github.com/iam-harshitha/raw2refined.git
    ```

2. **Navigate to the project directory**:

    ```bash
    cd raw2refined
    ```

3. **Install the required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

---

## How to Use

1. **Run the Streamlit app**:

    ```bash
    streamlit run app.py
    ```

2. **Upload CSV**: You will be prompted to upload a CSV file. After uploading, the app will display the first 5 rows, column data types, and counts of numerical and categorical columns.

3. **Missing Values**: If there are missing values, you can:
   - Drop columns with missing values.
   - Choose how to handle missing numerical and categorical values (e.g., fill with mean, median, or mode).

4. **Outlier Detection**: Detect outliers using the IQR method and choose whether to remove them.

5. **Handle Duplicates**: The app identifies duplicates and provides the option to remove them.

6. **Processed Data**: Once all preprocessing steps are complete, view and download the cleaned dataset.

---

## Technologies Used

- **Python**: The core programming language used.
- **Streamlit**: Used to create the web interface and host the interactive app.
- **Pandas**: For data manipulation and preprocessing.
- **Numpy**: For numerical operations.
- **Matplotlib & Seaborn**: For visualizing data and outlier detection.
- **Scikit-learn**: For scaling and preprocessing.

---

## Contributing

If you would like to contribute, feel free to create a pull request or open an issue to discuss improvements or bug fixes. Contributions are always welcome!

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
