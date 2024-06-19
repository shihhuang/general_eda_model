import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap
import math


class data_analysis():
    def __init__(self, data):
        # Read raw dataset
        print("==============================================================================================")
        if isinstance(data, str):
            print("Reading file from", data)
            self.raw_df = pd.read_csv(data)
        else:
            self.raw_df = data

        df_data_types = pd.DataFrame({"data_type": self.raw_df.dtypes})

        self.object_columns = list(df_data_types[df_data_types.data_type == "object"].index)
        self.non_object_columns = list(df_data_types[df_data_types.data_type != "object"].index)
        self.total_rows = self.raw_df.shape[0]

        print("Total number of rows read in:", self.total_rows)
        print("Non numeric columns:", self.object_columns)
        print("Numerical columns:", self.non_object_columns)
        print("Quick glance at the data: ")
        display(self.raw_df.head())
        print("==============================================================================================")

    def check_missing(self, num_rows=5):
        # Checking if there is missing values in each column
        missing_summary = pd.DataFrame({"num_rows_missing": self.raw_df.isna().sum(),
                                        "total_rows": self.total_rows
                                        })
        missing_summary["perc_missing"] = round(
            100.0 * missing_summary["num_rows_missing"] / missing_summary["total_rows"], 1)
        missing_columns = list(missing_summary[missing_summary["num_rows_missing"] > 0].index)
        print("Columns with nas")
        display(missing_summary.loc[missing_columns,])
        for column in missing_columns:
            display(self.raw_df[self.raw_df[column].isnull()].head(num_rows))

    def df_overview(self):
        """
        Function to have an overview on the data in a more concise and precise way
        :return:
        """
        # Default variables
        num_cols = 2  # number of columns in plot
        num_rows = 3  # number of rows in plot
        top_n = 10  # number of categoricals to plot
        num_col_in_table = 4

        #  =================================== Numeric Columns ===================================
        num_pages = math.ceil(len(self.non_object_columns) / 6)
        for page_num in range(num_pages):
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(8.67, 11.69))
            subset_df = self.raw_df[self.non_object_columns[page_num * 6:6 * (page_num + 1)]]
            fig.suptitle(f'Page {page_num + 1}: Histograms for Numeric Columns', fontsize=16)
            for i, category_to_plot in enumerate(subset_df.columns):
                row = i // num_cols
                col = i % num_cols
                ax = axs[row, col]
                ax.set_title("Histogram for " + category_to_plot, fontsize=12)
                ax.hist(subset_df[category_to_plot])

        #  =================================== Categorical Columns ===================================
        num_pages = math.ceil(len(self.object_columns) / 6)
        for page_num in range(num_pages):
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(8.67, 11.69))
            fig.subplots_adjust(hspace=0.5, wspace=0.3)
            fig.suptitle(f'Page {page_num + 1}: Histograms for Categorical Columns', fontsize=16)
            subset_df = self.raw_df[self.object_columns[page_num * 6:6 * (page_num + 1)]]

            for i, category_to_plot in enumerate(list(subset_df.columns)):
                row = i // num_cols
                col = i % num_cols
                ax = axs[row, col]
                # category_to_plot = "Job Level"
                categories = self.raw_df[category_to_plot].value_counts().index
                counts = self.raw_df[category_to_plot].value_counts().values
                total_counts = sum(counts)
                percentages = [i / total_counts for i in counts]
                ax.set_title("Top Categories for " + category_to_plot, fontsize=12)
                ax.barh(categories[0:top_n], counts[0:top_n], height=0.8)
                ax.invert_yaxis()
                ax.tick_params(axis='x', labelrotation=90)
                ax.tick_params(axis='y', labelsize=8)
                ax.set_yticklabels([textwrap.fill(label, width=12) for label in categories])  # Wrap labels

                rects = ax.patches
                for rect, label in zip(rects, percentages):
                    label_formatted = f"{label:.1%}"
                    y_value = rect.get_y() + rect.get_height() / 2
                    x_value = rect.get_width()
                    ax.text(
                        x_value, y_value, label_formatted, ha="center", va="center",
                        fontsize=10
                    )

        print("Total number of rows in the data: ", self.total_rows)
        for column in self.object_columns:
            print("Total number of unique values in", column, self.raw_df[column].nunique())
        print("Numerical values")
        num_numeric_cols = len(self.non_object_columns)
        num_tables = math.ceil(num_numeric_cols/num_col_in_table)
        for table_i in range(num_tables):
            table_df_all = pd.DataFrame()
            subset_cols = self.non_object_columns[table_i * num_col_in_table:num_col_in_table * (table_i + 1)]
            for col in subset_cols:
                df_summary = self.raw_df[col].describe()
                table_df_all = pd.concat([table_df_all,df_summary],axis=1)
            display(table_df_all)
