
## 1. Project Overview

This repository contains two Python scripts, `Visualizer.py` and `EXprctice.py`, which implement a command-line interface (CLI) tool for performing **Sales Data Analysis** and **Visualization**.


* Load data from a CSV file.
* Explore and clean the dataset.
* Perform various data manipulations (e.g., converting to NumPy, combining/splitting datasets, searching/sorting/filtering).
* Calculate aggregate functions, statistical analysis, and pivot tables.
* Generate various types of data visualizations (Bar, Box, Scatter, Histograms, Heatmaps, Pie, Stack Plots).



## 2. Requirements and Dependencies

The scripts require **Python 3.x** and the following external libraries:

| Library | Purpose |
| :--- | :--- |
| **pandas** | Core data manipulation and analysis. |
| **numpy** | Efficient numerical operations and array handling. |
| **matplotlib** | Foundational plotting library. |
| **seaborn** | Statistical data visualization (used heavily in `Visualizer.py`). |

## 3. Environment Setup

### 3.1. Install Python

Ensure you have Python 3.x installed on your system. You can download it from the official Python website or use a package manager like `brew` (macOS/Linux) or `choco` (Windows).

### 3.2. Install Dependencies

It is highly recommended to use a virtual environment to manage dependencies for this project.

1.  **Create a virtual environment (Optional but Recommended):**
    ```bash
    python -m venv venv
    ```

2.  **Activate the virtual environment:**
    * **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

3.  **Install the required libraries using pip:**
    ```bash
    pip install pandas numpy matplotlib seaborn
    ```

## How to run code:

====== Main Menu =====
1. Load Dataset
2. Explore Data
3. Perform DataFrame Operations
...
First Step:
You must select "1. Load Dataset" and provide the file path to a valid CSV file containing your sales data (e.g., sales_data.csv).

Interactive Use:
Navigate the menus by entering the corresponding number (1-8) and following the on-screen prompts for column names, values, and visualization settings.

Saving Plots (Visualizer.py):
After generating a plot (Menu 6), you must select "7. Save and Display Last Plot" from the Main Menu to save the visualization to a file (e.g., sales_bar.png) and close the figure.






