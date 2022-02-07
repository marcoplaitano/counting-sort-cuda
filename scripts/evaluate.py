"""
File:   evaluate.py
Brief:  This script reads the measured performances data and creates CSV tables
        and plots to visualize the Speedup and Efficiency of each measure.
Author: MarcoPlaitano
Date:   19 Jan 2022

COUNTING SORT CUDA
Parallelize and Evaluate Performances of "Counting Sort" Algorithm, by using
CUDA.

Copyright (C) 2022 Plaitano Marco

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

from os import path, scandir, walk
from sys import argv
from decimal import Decimal
from pandas import read_csv
from scipy.stats import norm
from prettytable import PrettyTable



def my_round(num: float) -> float:
    """Round up the given float number to its first 5 decimal points."""
    return float(round(Decimal(num), 5))



def calculate_means(files: list) -> dict:
    """Calculate, for every file, the mean of every parameter."""
    # Dictionary in which keys are filenames and values are other dictionaries
    # containing: parameter names as keys and their means as values.
    all_means = {}

    # Parameters to calculate the mean of.
    columns = ["time_kernel", "time_sort"]

    for file in files:
        file_mean = {}
        content = read_csv(file, sep=';')

        for col in columns:
            curr_data = content[col]
            # Mean and Standard Deviation
            # .norm generates a Normal Continuos Distribution
            # .fit generates the MLE (Maximum Likelihood Estimation) for the
            # given data by minimizing the negative log-likelihood function.
            # The return values are location and scale parameters. For a normal
            # distribution the location is its mean.
            mean, std = norm.fit(curr_data)
            # Remove values that are too unlikely; in other words, only keep
            # values inside the range  [mean - std, mean + std]
            curr_data = content[(content[col] > (mean - std)) &
                                (content[col] < (mean + std))][col]
            if len(curr_data) == 0:
                mean = my_round(0.0)
            else:
                mean = norm.fit(curr_data)[0]
            file_mean[col] = my_round(mean)

        all_means[file] = file_mean

    return all_means



def parse_file_name(filename: str) -> tuple:
    """Get measure info (size, grid_size, block_size) from file name.

    An example of a filename is: Bxx_Syy.csv, where:
     - xx   is the Block size
     - yy   is the Size of the Array."""
    filename = path.basename(filename)

    # Extract block size.
    bs_start = filename.index('B') + 1
    bs_end = filename.index('_', bs_start)
    block_size = int(filename[bs_start : bs_end])

    # Extract size.
    size_start = filename.index('S') + 1
    size_end = filename.index('.', size_start)
    size = int(filename[size_start : size_end])

    # Calculate grid size.
    if block_size == 0:
        grid_size = 0
    else:
        grid_size = int((size - 1) / block_size + 1)

    return size, grid_size, block_size



def create_row(file: str, size: int, grid_size: int, block_size: int,
               means: dict) -> list:
    """Create a list containing all the information taken from the arguments."""
    row = []

    row.append(size)
    row.append(grid_size)
    row.append(block_size)

    # All the means of the measure's parameters.
    for col in means[file]:
        row.append(means[file][col])

    return row



def make_table(root_dir: str, files: list, means: dict) -> None:
    """Create a table storing, for every type of test, the mean of the results.

    A new table will be created for each problem size.
    """

    fields = ["Size", "Grid Size", "Block Size", "Time Kernel", "Time Sort"]
    rows = []

    # All subdirectories in the output's root directory.
    subdirs = sorted([d.path for d in scandir(root_dir) if d.is_dir()])

    for subdir in subdirs:
        # These are all the files in the current subdirectory.
        curr_files = sorted([f for f in files if f.find(subdir + "/") >= 0])

        # For every file in the directory create a new row to insert in the
        # current table.
        for file in curr_files:
            size, grid_size, block_size = parse_file_name(file)

            # Add all details in a list.
            row = create_row(file, size, grid_size, block_size, means)
            rows.append(row)

        # When all the rows have been built, write the table onto a file.
        write_table(fields, rows, subdir)

        rows.clear()



def write_table(fields: list, rows: list, directory: str) -> None:
    """Write a table with fields and rows onto the given file."""
    table = PrettyTable()
    table.field_names = fields
    table.add_rows(rows)
    # Write table in standard CSV format, separator is ','.
    with open(directory + "/table.csv", "w", encoding="UTF-8") as file:
        file.write(table.get_csv_string())



def main() -> None:
    """Main function."""

    if len(argv) < 2:
        print("usage: python3 ./evaluate.py dir")
        print("'dir' is the directory containing the output produced by "
              "measures.sh")
        exit(1)
    directory = argv[1]

    # Get all files contained in the given directory
    files = []
    for root, dirs, files_found in walk(directory):
        for file in files_found:
            if file.endswith(".csv") and file.find("table") == -1:
                files.append(path.join(root, file))
    files = sorted(files)

    if len(files) == 0:
        print("No output files found. Exiting the script.")
        exit(1)

    print("Calculating means...")
    means = calculate_means(files)

    print("Creating tables and plots...")
    make_table(directory, files, means)



if __name__ == "__main__":
    main()
