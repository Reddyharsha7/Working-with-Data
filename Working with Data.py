import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

try:
    # Load the Iris dataset from seaborn
    iris = sns.load_dataset('iris')
    print("Iris dataset loaded successfully.")

    # Display the first few rows of the dataset
    print(iris.head())

    # ---- Data Manipulation ----

    # 1. Filtering: Get only the 'setosa' species
    setosa_data = iris[iris['species'] == 'setosa']
    print("\nFiltered Data for 'Setosa' species:")
    print(setosa_data.head())

    # 2. Grouping and Aggregating: Group by species and calculate the mean of each numeric column
    grouped_data = iris.groupby('species').mean()
    print("\nGrouped Data by Species (Mean of Numeric Columns):")
    print(grouped_data)

    # 3. Creating a New Column: Petal Length to Petal Width Ratio
    iris['petal_ratio'] = iris['petal_length'] / iris['petal_width']
    print("\nData with New Column (Petal Length to Petal Width Ratio):")
    print(iris[['species', 'petal_length', 'petal_width', 'petal_ratio']].head())

    # ---- Data Visualization ----

    # 1. Scatter Plot: Sepal Length vs Sepal Width
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='sepal_length', y='sepal_width', data=iris, hue='species')
    plt.title('Sepal Length vs Sepal Width')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.legend(title='Species')
    plt.savefig('sepal_length_vs_sepal_width.png')  # Save the plot
    plt.show()

    # 2. Box Plot: Sepal Length by Species
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='species', y='sepal_length', data=iris)
    plt.title('Box Plot of Sepal Length by Species')
    plt.xlabel('Species')
    plt.ylabel('Sepal Length')
    plt.savefig('sepal_length_by_species_boxplot.png')  # Save the plot
    plt.show()

    # 3. Pair Plot: Visualizing relationships between multiple variables
    sns.pairplot(iris, hue='species')
    plt.savefig('pairplot_iris_dataset.png')  # Save the plot
    plt.show()

    print("\nData analysis completed successfully.")
except Exception as e:
    print("An error occurred:", e)
