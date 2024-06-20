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
                main_file = csv.reader(combined_file)
                featureHeader = next(main_file)
                total_rows.append(featureHeader)
                total_rows.extend(row for row in main_file)

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

        directory = 'impala_analysis_files'
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created.")
        else:
            print(f"Directory '{directory}' already exists.")

        combined_csv_path = 'impala_analysis_files/combined_output.csv'
        with open(combined_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(total_rows)

        print("All CSV files have been combined into 'combined_output.csv'")


def pair_plot(combined_file_path,dirPath):
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
    plt.savefig(dirPath + '/pairplot_mean' + '.png')
    plt.close()

    sns.pairplot(df_pairplot_median, y_vars=median_column,
                 x_vars=df_pairplot_median.columns.difference([mean_column, median_column]))
    plt.show()
    plt.savefig(dirPath + '/pairplot_median' + '.png')
    plt.close()


def correlation_analysis(combinedFilePath):
    df = pd.read_csv(combinedFilePath)
    median_column = df.columns[-1]
    mean_column = df.columns[-2]
    correlation_with_mean = df.drop(columns=[median_column,mean_column]).corrwith(df[mean_column])
    correlation_with_median = df.drop(columns=[mean_column, median_column]).corrwith(df[median_column])

    print('correlation with median:\n',correlation_with_median.sort_values(ascending=False))
    print('correlation with mean:\n',correlation_with_mean.sort_values(ascending=False))

    return correlation_with_mean, correlation_with_median


if __name__ == "__main__":
    combine_files = False
    dirPath = '/data/adatta14/PycharmProjects/ni-scheduling/simulator/impala_analysis_files'
    initialPath = '/data/adatta14/PycharmProjects/ni-scheduling/simulator/impala_analysis_files/combined_output.csv'
    if combine_files:
       combine_all_csv_files(dirPath, initialPath, True)
       combined_csv_path = '/data/adatta14/PycharmProjects/ni-scheduling/simulator/impala_analysis_files/combined_output.csv'
       pair_plot(combined_csv_path,dirPath)
    else:
       correlation_analysis(initialPath)
