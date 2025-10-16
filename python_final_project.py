import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class SalesDataAnalyzer:
    # --- Initialization --- 
    def __init__(self, file_path=None):
        self.df = pd.DataFrame() # To store dataframe
        self.numpy_array = np.array([]) # To store Numpy array
        self.file_path = file_path # To store file 
        self.last_plot = None  # To store the last generated figure object
        if file_path:
            self.load_data(file_path)

    # --- Destructor ---
    def __del__(self):
        print("\nAnalyzer object destroyed. Cleanup completed.")
    
    # --- load csv file ---
    def load_data(self, file_path=None):
        if not file_path:
            file_path = input("Enter CSV file path: ").strip()
        try:
            self.df = pd.read_csv(file_path)
            self.numpy_array = self.df.to_numpy() # convert dataframe into a Numpy array
            print("\nFile loaded successfully!")
        except FileNotFoundError:
            print("\nFile not found.")
        except Exception as e:
            print(f"\nError: {e}")

    # --- Explore data ---
    def explore(self):
        if self.df.empty:
            print("load data first.")
            return

        while True:
            print("\n=== Explore Data ===")
            print("1. Display First 5 rows")
            print("2. Display Last 5 rows")
            print("3. Display Column Names")
            print("4. Display Data Types")
            print("5. Display Basic Info")
            print("6. Go Back")
            choice = input("Enter your choice: ").strip()

            if choice == "1":  print(self.df.head())
            elif choice == "2": print(self.df.tail())
            elif choice == "3": print(list(self.df.columns))
            elif choice == "4":  print(self.df.dtypes)
            elif choice == "5": self.df.info()
            elif choice == "6":  break
            else: print("Invalid choice!")

    # --- Handle missinf value --- 
    def clean_data(self):
        """Handle missing values and clean the dataset."""
        if self.df.empty:
            print("Please load the data first.")
            return
    
        print("\n--- Checking for missing values ---")
        print(self.df.isnull().sum())
    
        rows_before = len(self.df)
        self.df.dropna(subset=['Fare', 'Age'], inplace=True)
        rows_after = len(self.df)
        print(f"\nRemoved {rows_before - rows_after} rows with missing values in key columns.")
    
        self.df.fillna({'Fare': 'Unknown', 'Age': 'Unknown'}, inplace=True)
    
        print("\nData cleaning completed.")

    # --- Convert numpy ---
    def convert_numpy(self):
        if self.df.empty:
            print("Load data first.")
            return
    
        try:
            # -- Convert columns --
            cols = input("Enter column names to convert to NumPy (comma-separated, or leave blank for all): ").strip()
            if cols:
                cols = [c.strip() for c in cols.split(',')]
                self.numpy_array = self.df[cols].values
            else:
                self.numpy_array = self.df.values
            print("\nData successfully converted into a NumPy array.") 
    
            # -- indexing --
            row_index = input("Enter row index to access (leave blank to skip indexing): ").strip()
            col_index = input("Enter column index to access (leave blank to skip indexing): ").strip()
            if row_index.isdigit() and col_index.isdigit():
                row_index = int(row_index)
                col_index = int(col_index)
                print(f"Value at row {row_index}, column {col_index}: {self.numpy_array[row_index, col_index]}")
            else:
                print("Skipping indexing.")
    
            # -- slicing --
            row_slice = input("Enter row slice (start:end, leave blank for all rows): ").strip()
            col_slice = input("Enter column slice (start:end, leave blank for all columns): ").strip()
    
            def parse_slice(s):
                if not s:
                    return slice(None)  # select all
                if ':' in s:
                    start, end = s.split(':')
                    start = int(start.strip()) if start.strip() else None
                    end = int(end.strip()) if end.strip() else None
                    return slice(start, end)
                else:
                    return int(s)  # single index
    
            row_slice_obj = parse_slice(row_slice)
            col_slice_obj = parse_slice(col_slice)
    
            print(f"\nSliced array:\n{self.numpy_array[row_slice_obj, col_slice_obj]}")
    
        except KeyError:
            print("Columns are not in the dataset.")
        except Exception as e:
            print(f"Error: {e}")

                # --- Mathematical opreation ---
    def mathematical_operations(self):
        """Perform mathematical operations on the DataFrame."""
        if self.df.empty:
            print("Please load data first.")
            return

        try:
            sales_col = input("Enter the column name for Sales: ").strip()
            profit_col = input("Enter the column name for Profit: ").strip()

            if sales_col not in self.df.columns or profit_col not in self.df.columns:
                print("Error: One or both columns not found in the DataFrame.")
                return

            self.df['Tax'] = self.df[sales_col] * 0.05
            self.df['Cost'] = self.df[sales_col] - self.df[profit_col] - self.df['Tax']

            print("\nColumns 'Tax' and 'Cost' added.")
            print(self.df[[sales_col, profit_col, 'Tax', 'Cost']].head())

        except Exception as e:
            print(f"Error performing mathematical operations: {e}")


    # --- Combine data ---
    def combine_data(self):
        file_path = input("Enter CSV path of dataset to combine: ").strip()
        try:
            other_df = pd.read_csv(file_path)
            self.df = pd.concat([self.df, other_df], ignore_index=True)
            print(f"Datasets combined. Total rows: {len(self.df)}")
        except Exception as e:
            print(f"Error: {e}")

    # --- Split data ---
    def split_data(self):
        if self.df.empty:
            print("Load data first.")
            return

        col = input("Enter column name to split by: ").strip()
        if col not in self.df.columns:
            print(f"Column '{col}' not found.")
            return
            
        val = input(f"Enter value of '{col}' to filter first dataset: ").strip()
        if pd.api.types.is_numeric_dtype(self.df[col]):
            try:
                val = float(val)
            except ValueError:
                print(f"Invalid numeric value '{val}' for column '{col}'.")
                return

        df1 = self.df[self.df[col] == val].copy()
        df2 = self.df[self.df[col] != val].copy()
        print(f"Dataset 1 ('{col}' == {val}) rows: {len(df1)}")
        print(f"Dataset 2 ('{col}' != {val}) rows: {len(df2)}")
        return df1, df2

    # --- Search & Sort & Filter data ---
    def search_sort_filter(self):
        if self.df.empty:
            print("Load data first.")
            return

        # -- Search -- 
        col = input("Enter column name to search: ").strip()
        val = input(f"Enter value to search in '{col}': ").strip()
        results = self.df[self.df[col].astype(str).str.contains(val, case=False, na=False)]
        print(f"\nFound {len(results)} rows where '{col}' contains '{val}':")
        print(results.head())

        # -- Sort --
        sort_col = input("\nEnter column name to sort by: ").strip()
        if sort_col in self.df.columns:
            sorted_df = self.df.sort_values(by=sort_col, ascending=False)
            print(f"\nTop 5 rows sorted by '{sort_col}' (descending):")
            print(sorted_df.head())
        else:
            print(f"Column '{sort_col}' not found.")

        # -- Filter --
        filter_col = input("\nEnter column name to filter by (numeric): ").strip()
        if filter_col in self.df.columns:
            try:
                filter_val = float(input(f"Enter value for '{filter_col}': ").strip())
                filtered_df = self.df[self.df[filter_col] >= filter_val]
                print(f"\nData filtered where '{filter_col}' >= {filter_val}:")
                print(filtered_df.head())
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
            except TypeError:
                print(f"Cannot perform numeric filter on non-numeric column '{filter_col}'.")
        else:
            print(f"Column '{filter_col}' not found.")

    # --- Aggregate function ---
    def aggregate_functions(self):
        if self.df.empty:
            print("Load data first.")
            return

        col = input("Enter column name to aggregate (e.g., Sales): ").strip()
        if col not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[col]):
            print(f"Column '{col}' must be a numeric column in the DataFrame.")
            return

        print(f"\nAggregate Functions for '{col}':")
        print(f"Sum: {self.df[col].sum():.2f}")
        print(f"Mean: {self.df[col].mean():.2f}")
        print(f"Count: {self.df[col].count()}")

    # --- Statistical Operation ---
    def statistical_analysis(self):
        if self.df.empty:
            print("Load data first.")
            return

        col = input("Enter column name for statistical analysis: ").strip()
        if col not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[col]):
            print(f"Column '{col}' must be a numeric column in the DataFrame.")
            return

        print(f"\nStatistical Analysis for '{col}':")
        print(f"Standard Deviation: {self.df[col].std():.2f}")
        print(f"Variance: {self.df[col].var():.2f}")
        print("Quantiles:")
        print(self.df[col].quantile([0.25, 0.5, 0.75]))

    # --- Pivot table ---
    def create_pivot_table(self):
        if self.df.empty:
            print("Load data first.")
            return
        
        try:
            index_col = input("Enter index columns for pivot (comma-separated): ").strip().split(',')
            values_col = input("Enter value columns for pivot (comma-separated): ").strip().split(',')
            agg_func = input("Enter aggregation function (sum, mean, count): ").strip()

            pivot = pd.pivot_table(self.df, values=values_col, index=index_col, aggfunc=agg_func)
            print("\nPivot Table:")
            print(pivot)
        except Exception as e:
            print(f"Error: {e}")

    # --- visualize data ---
    def visualize_data(self):
        if self.df.empty:
            print("Load data first.")
            return
        
        # Set a theme for all plots
        sns.set_theme(style="whitegrid")
        
        while True:
            print("\n==== Visualization Menu ====")
            print("1. Bar Plot")
            print("2. Box Plot")
            print("3. Scatter Plot")
            print("4. Histogram ")
            print("5. Heatmap")
            print("6. Pie Chart")
            print("7. Stack Plot")
            print("8. Go Back")
            choice = input("Enter plot type number: ").strip()
            
            if choice == "8": break
            
            if choice not in [str(i) for i in range(1, 9)]:
                print("Invalid choice.")
                continue

            try:
                # Store the created figure object
                self.last_plot = plt.figure(figsize=(10, 6))
                
                if choice == "1": # Bar Plot
                    x_col = input("Enter categorical column for X-axis (e.g., Region): ").strip()
                    y_col = input("Enter numeric column for Y-axis (e.g., Sales): ").strip()
                    sns.barplot(data=self.df, x=x_col, y=y_col, estimator=sum, errorbar=None, palette='viridis', hue=x_col, legend=False)
                    plt.title(f'Total {y_col} by {x_col}')
                    plt.xticks(rotation=45)
        
                elif choice == "2": # Box Plot
                    x_col = input("Enter categorical column for X-axis (e.g., Category): ").strip()
                    y_col = input("Enter numeric column for Y-axis (e.g., Sales): ").strip()
                    sns.boxplot(data=self.df, x=x_col, y=y_col, hue=x_col, palette='pastel', legend=False)
                    plt.title(f'Distribution of {y_col} by {x_col}')
                    plt.xticks(rotation=45)
                    
                elif choice == "3": #  Scatter Plot 
                    x_col = input("Enter X-axis column (e.g., Sales): ").strip()
                    y_col = input("Enter Y-axis column (e.g., Profit): ").strip()
                    hue_col = input("Enter categorical column for color (e.g., Region): ").strip()
                    sns.scatterplot(data=self.df, x=x_col, y=y_col, hue=hue_col, palette='deep', alpha=0.7)
                    plt.title(f'{y_col} vs {x_col} by {hue_col}')
        
                elif choice == "4": # Histogram 
                    col = input("Enter numeric column for histogram (e.g., Sales): ").strip()
                    bins = int(input("Enter number of bins (e.g., 30): "))
                    sns.histplot(self.df[col], bins=bins, kde=True, color='purple')
                    plt.title(f'Distribution of {col}')
                
                elif choice == "5": #  Heatmap
                    numeric_df = self.df.select_dtypes(include=np.number)
                    corr_matrix = numeric_df.corr()
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
                    plt.title('Correlation Matrix of Numeric Columns')

                elif choice == "6": # Pie Chart
                    col = input("Enter column for pie chart (categorical): ").strip()
                    counts = self.df[col].value_counts()
                    top_counts = counts.nlargest(10)
                    plt.pie(top_counts, labels=top_counts.index, autopct='%1.1f%%', startangle=140)
                    plt.title(f'Distribution of {col}')
                    plt.axis('equal')

                elif choice == "7": # Stack Plot
                    print("\n--- Stack Plot Setup ---")
                    time_col = input("Enter time/date column for X-axis (e.g., Date/Year): ").strip()
                    category_col = input("Enter category column to stack (e.g., Region, Product): ").strip()
                    value_col = input("Enter numeric column for values (e.g., Sales, Profit): ").strip()

                    pivot_data = self.df.pivot_table(
                        index=time_col,
                        columns=category_col,
                        values=value_col,
                        aggfunc='sum'
                    ).fillna(0)
                    
                    if pivot_data.empty or len(pivot_data.columns) == 0:
                         print("Error: Could not create pivot table.")
                         plt.close(self.last_plot) # Close the figure created earlier
                         self.last_plot = None
                         continue

                    plt.stackplot(
                        pivot_data.index, 
                        pivot_data.values.T, 
                        labels=pivot_data.columns, 
                        alpha=0.8
                    )

                    plt.title(f'{value_col} Stacked by {category_col} Over {time_col}')
                    plt.xlabel(time_col)
                    plt.ylabel(f"Total {value_col}")
                    plt.legend(title=category_col, loc='upper left')
                    plt.xticks(rotation=45)

                self.last_plot.tight_layout()
                print("\nPlot successfully generated. Use Save Plot (Menu 7) to save it.")

            except KeyError as e:
                print(f"Error: Column not found. Check the column name you entered: {e}")
                plt.close(self.last_plot)
                self.last_plot = None
            except Exception as e:
                print(f"Error: {e}")
                plt.close(self.last_plot)
                self.last_plot = None


    # ---  save visualization ---
    def save_visualization(self):
        if self.last_plot is None:
            print("\nNo plot has been generated yet. Please choose a visualization option first (Menu 6).")
            return

        try:
            # Get the filename from the user
            file_name = input("\nEnter filename to save plot (e.g., plot.png): ").strip()

            if file_name:
                self.last_plot.savefig(file_name)
                print(f"Plot saved successfully as {file_name}")
            else:
                 print("\nSkipped saving the plot.")
                
            plt.close(self.last_plot)
            self.last_plot = None 
            print("Figure closed and memory released.")
            
        except Exception as e:
            print(f"Error saving or closing plot: {e}")
            if self.last_plot:
                 plt.close(self.last_plot)
            self.last_plot = None 


def main():
    analyzer = SalesDataAnalyzer()

    while True:
        print("\n====== Main Menu =====")
        print("1. Load Dataset")
        print("2. Explore Data")
        print("3. Perform DataFrame Operations")
        print("4. Clean and Handle Missing Data")
        print("5. Generate Descriptive Statistics")
        print("6. Data Visualization")
        print("7. Save and Display Last Plot")
        print("8. Exit")

        choice = input("Enter your choice: ").strip()

        if choice == "1": analyzer.load_data()
        elif choice == "2": analyzer.explore()
        elif choice == "3":
            while True:
                print("\n--- Operations Menu ---")
                print("1. Convert to NumPy")
                print("2. Mathematical Operations")
                print("3. Combine Datasets")
                print("4. Split Dataset")
                print("5. Search / Sort / Filter")
                print("6. Aggregate Functions")
                print("7. Back to Main Menu")
                op = input("Enter operation: ").strip()
                
                if op == "1": analyzer.convert_numpy()
                elif op == "2": analyzer.mathematical_operations()
                elif op == "3": analyzer.combine_data()
                elif op == "4": analyzer.split_data()
                elif op == "5": analyzer.search_sort_filter()
                elif op == "6": analyzer.aggregate_functions()
                elif op == "7": break
                else: print("Invalid choice.")
        
        elif choice == "4": analyzer.clean_data()
        elif choice == "5":
            while True:
                print("\n--- Statistics Menu ---")
                print("1. Statistical Analysis")
                print("2. Pivot Table")
                print("3. Back to Main Menu")
                stat = input("Enter choice: ").strip()
                
                if stat == "1": analyzer.statistical_analysis()
                elif stat == "2": analyzer.create_pivot_table()
                elif stat == "3": break
                else: print("Invalid choice.")
                
        elif choice == "6": analyzer.visualize_data()
        elif choice == "7": analyzer.save_visualization()
        elif choice == "8":
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()

