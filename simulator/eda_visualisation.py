import seaborn as sns
import matplotlib.pyplot as plt
import csv
import os
import pandas as pd


def combine_all_csv_files(dirPath, initialPath, initial_file):
    # list for data
    total_rows = []
    # list for headers
    featureHeader = None

    if not initial_file:
        # List all CSV files and get their paths
        # Check if combined_output.csv exists
        if os.path.exists(initialPath):
            with open(initialPath, 'r', newline='') as combined_file:
                combined_reader = csv.reader(combined_file)
                featureHeader = next(combined_reader)
                total_rows.append(featureHeader)
                total_rows.extend(row for row in combined_reader)

        # List all CSV files and get their paths, excluding combined_output.csv
        csv_files = [os.path.join(dirPath, file) for file in os.listdir(dirPath) if
                     file.endswith('csv') and file != 'combined_output.csv']

        # If no CSV files found, return immediately
        if not csv_files:
            print("No CSV files found in the directory.")
            return None

        # Find the most recently created CSV file
        last_file = max(csv_files, key=os.path.getctime)

        # Read the most recent file
        with open(last_file, 'r', newline='') as csvfile:
            row_data = csv.reader(csvfile)
            next(row_data, None)
            total_rows.extend(row for row in row_data)

        # Write the updated data to the combined CSV file
        with open(initialPath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(total_rows)

        print(f"Data from the most recent file '{os.path.basename(last_file)}' has been added to 'combined_output.csv'")

    else:
        for file in os.listdir(dirPath):
            if file.endswith('csv'):
                filePath = os.path.join(dirPath, file)
                with open(filePath, 'r', newline='') as csvfile:
                    row_data = csv.reader(csvfile)
                    if featureHeader is None:
                        featureHeader = next(row_data)
                        total_rows.append(featureHeader)
                    else:
                        next(row_data)
                    total_rows.extend(row for row in row_data)

        directory = 'analysis_files'
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created.")
        else:
            print(f"Directory '{directory}' already exists.")

        combined_csv_path = 'analysis_files/combined_output.csv'
        with open(combined_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(total_rows)

        print("All CSV files have been combined into 'combined_output.csv'")


def correlation_and_pair_plot(combined_file_path):
    df = pd.read_csv(combined_file_path)
    # Get the name of the last column
    mean_column = df.columns[-2]
    median_column = df.columns[-1]

    # Create a new DataFrame with only the columns you're interested in
    df_pairplot_mean = df[df.columns.difference([mean_column, median_column])]
    df_pairplot_mean[mean_column] = df[mean_column]

    df_pairplot_median = df[df.columns.difference([mean_column, median_column])]
    df_pairplot_median[median_column] = df[median_column]

    # Create the pairplots
    sns.pairplot(df_pairplot_mean, y_vars=mean_column,
                 x_vars=df_pairplot_mean.columns.difference([mean_column, median_column]))
    plt.show()
    plt.savefig('analysis_files/pairplot_mean' + '.png')
    plt.close()

    sns.pairplot(df_pairplot_median, y_vars=median_column,
                 x_vars=df_pairplot_median.columns.difference([mean_column, median_column]))
    plt.show()
    plt.savefig('analysis_files/pairplot_median' + '.png')
    plt.close()


if __name__ == "__main__":
    dirPath = '/data/adatta14/PycharmProjects/ni-scheduling/simulator/analysis_files'
    initialPath = '/data/adatta14/PycharmProjects/ni-scheduling/simulator/analysis_files/combined_output.csv'
    combine_all_csv_files(dirPath, initialPath, False)
    combined_csv_path = '/data/adatta14/PycharmProjects/ni-scheduling/simulator/analysis_files/combined_output.csv'
    correlation_and_pair_plot(combined_csv_path)
