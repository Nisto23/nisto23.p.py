import sys
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure necessary packages are installed
def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Correct package name for scikit-learn
install_and_import("scikit-learn")
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
def load_and_explore_iris():
    try:
        # Load Iris dataset from sklearn
        iris = load_iris(as_frame=True)
        df = iris.frame

        # Add target names for clarity
        df['species'] = df['target'].map({i: name for i, name in enumerate(iris.target_names)})
        df.drop(columns='target', inplace=True)

        # Display the first few rows
        print("First few rows of the dataset:")
        print(df.head())

        # Check dataset structure
        print("\nDataset Info:")
        print(df.info())

        # Check for missing values
        print("\nMissing Values:")
        print(df.isnull().sum())

        return df

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# Task 2: Basic Data Analysis
def analyze_data(df):
    try:
        # Basic statistics
        print("\nBasic Statistics:")
        print(df.describe())

        # Grouping by species and calculating means
        group_means = df.groupby('species').mean()
        print("\nMean values grouped by species:")
        print(group_means)

        # Observations
        print("\nObservations:")
        print("Setosa species has smaller sepal and petal dimensions compared to other species.")
        print("Versicolor and Virginica have overlapping sepal lengths but distinct petal widths.")

        return group_means

    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        return None

# Task 3: Data Visualization
def visualize_data(df, group_means):
    try:
        # Line chart (Hypothetical trends over index)
        plt.figure(figsize=(10, 5))
        for species, data in df.groupby('species'):
            plt.plot(data.index, data['sepal length (cm)'], label=species)
        plt.title("Sepal Length Trend by Species")
        plt.xlabel("Index")
        plt.ylabel("Sepal Length (cm)")
        plt.legend()
        plt.show()

        # Bar chart
        plt.figure(figsize=(8, 5))
        sns.barplot(x=group_means.index, y=group_means['petal length (cm)'])
        plt.title("Average Petal Length by Species")
        plt.xlabel("Species")
        plt.ylabel("Average Petal Length (cm)")
        plt.show()

        # Histogram
        plt.figure(figsize=(8, 5))
        sns.histplot(df['sepal width (cm)'], kde=True, bins=15)
        plt.title("Distribution of Sepal Width")
        plt.xlabel("Sepal Width (cm)")
        plt.ylabel("Frequency")
        plt.show()

        # Scatter plot
        plt.figure(figsize=(8, 5))
        sns.scatterplot(
            x=df['sepal length (cm)'], 
            y=df['petal length (cm)'], 
            hue=df['species']
        )
        plt.title("Sepal Length vs Petal Length by Species")
        plt.xlabel("Sepal Length (cm)")
        plt.ylabel("Petal Length (cm)")
        plt.legend(title="Species")
        plt.show()

    except Exception as e:
        print(f"An error occurred during visualization: {e}")

# Main script execution
iris_df = load_and_explore_iris()
if iris_df is not None:
    group_means = analyze_data(iris_df)
    if group_means is not None:
        visualize_data(iris_df, group_means)
