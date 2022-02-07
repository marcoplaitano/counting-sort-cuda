#!/bin/bash

# File:   measures.sh
# Brief:  This script compiles and runs the C program multiple times to measure
#         its execution time taking into account different parameters.
# Author: Marco Plaitano
# Date:   19 Jan 2022
#
# COUNTING SORT CUDA
# Parallelize and Evaluate Performances of "Counting Sort" Algorithm, by using
# CUDA.
#
# Copyright (C) 2022 Plaitano Marco
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


function show_guide {
    echo $"NAME
    Measures

SYNOPSIS
    measures.sh [OPTION]...

DESCRIPTION
    This script compiles and runs both the serial and parallel versions of the C
    program in the src/ directory to measure its execution time. Various tests
    are carried out taking into account different values of the following
    parameters:
        - Problem size
        - Block size.

    Each combination is tried out multiple times; when the tests are completed,
    the 'evaluate.py' script is launched to estimate means and draw tables of
    the newly generated results.

OPTIONS
    -h, --help
        Show this guide and exit.

    -n N, --num-measures N, -n=N, --num-measures=N
        Run every measure N times. (default is 20)

    -d DIR, --dir DIR
        Write all the output in the DIR directory of your choice instead of the
        default location './output/'.

    --no-table
        Do not run the Python script to generate the tables." | more -d
}


# Delete all temporary files before exiting.
# Argument $1 is the exit code.
function safe_exit {
    echo $'\n'"Deleting temporary output..."
    [[ -f "$tmp_file" ]]    && rm "$tmp_file"
    [[ -d "$project_dir"/scripts/__pycache__ ]] && \
    rm -r "$project_dir"/scripts/__pycache__
    echo "Exiting script."
    exit $1
}

# Redirect to the safe_exit function if user presses 'CTRL+C'.
trap "safe_exit 0" SIGINT


# Print error message and interrupt the execution of the script.
function raise_error {
    [[ -n "$1" ]] && echo "$1" || echo "Error."
    safe_exit 1
}


# Compile source file(s) via makefile.
function compile {
    target="$1"
    make -C "$project_dir" clean > /dev/null 2>&1
    make -C "$project_dir" "$target" > /dev/null
    [[ $? != 0 ]] && raise_error
}


# Measure the execution time and save the results on a file.
function measure_time {
    # First line in CSV file declares the columns format.
    echo "size;gridSize;blockSize;time_kernel;time_sort" > "$output_file"

    # Show initial 0% progress.
    printf "\r[  0/%d   0%%]" $num_measures

    for (( i=1; i<=$num_measures; i++ )); do
        # Time the execution of the program and save the result in a temp file.
        "$exec_file" "${exec_args[@]}" > "$tmp_file" 2>&1

        # Exit the script if the execution produced an error.
        if [[ $? != 0 ]]; then
            raise_error "An error occurred during the execution of the program."
        fi

        # Normalize temporary output and save it.
        cat "$tmp_file" | sed -e 's/,/./g' >> "$output_file"

        # Show current progress.
        printf "\r[%3d/%d %3d%%] " $i $num_measures $(($i*100 / $num_measures))
        printf "=%.0s" $(seq -s " " 1 $(( $i * 50 / $num_measures)) )
    done
    echo
}


# Return a string representing a number with leading zeros.
# Argument $1 is the number to modify.
# Argument $2 is the maximum value to use as scale when deciding the number of
# zeros to add.
# Example:
#   $1 = 5, $2 = 10  produces the output:  05
#   $1 = 5, $2 = 100 produces the output: 005
function add_leading_zeros {
    printf "%0*d" ${#2} $1
}



# Parse command line arguments.
while [[ -n $1 ]]; do
    case $1 in
        -h | --help)
            show_guide
            exit 0 ;;
        -n | --num-measures)
            num_measures=$2
            [[ -z $num_measures ]] && raise_error "No number of measures given."
            shift ; shift ;;
        -d | --dir)
            output_dir="$2"
            [[ -z $output_dir ]] && raise_error "No output directory given."
            shift ; shift ;;
        --no-table)
            no_table=1
            shift ;;
        *)
            raise_error "Argument '$1' not recognized." ;;
    esac
done

# Check that the argument is actually a positive integer
if [[ -n $num_measures ]] && [[ ! $num_measures =~ ^[0-9]+$ ]]; then
    raise_error "Not a valid number of measures."
fi


# Determine root project directory based on whether this script has been
# launched from there or from the scripts/ subdirectory.
project_dir=$(pwd)
[[ $(pwd) == *scripts ]] && project_dir=${project_dir%"scripts"}

# Set number of measures to perform, if not given.
num_measures=${num_measures:="20"}

# Different sizes for the array in the program.
sizes=(512000 102400000)

# Threads' block sizes to execute the program with. When 0 the serial version
# is executed.
block_sizes=(0 64 256 1024)

# If not given set directory in which to store measurements results.
output_dir=${output_dir:="$project_dir/output"}
# Create the directory if it does not exist yet.
[[ ! -d "$output_dir" ]] && mkdir -p "$output_dir"
# Delete any previously generated content (if user agrees).
if [[ "$(ls "$output_dir")" ]]; then
    echo "Warning: the output directory '""$output_dir""' is not empty."
    read -p "Delete previous output (all contents) from it? [y;N] " answer
    case $answer in
        [yY] | [yY][eE][sS])
            rm -Ir "$output_dir"/* ;;
        *)
            echo "Not deleting." ;;
    esac
fi

# File in which to store temporary output
tmp_file="$output_dir"/.temp.txt

# Executable file compiled from C source that has to be launched.
exec_file="$project_dir/bin/main.out"


for block_size in ${block_sizes[@]}; do
    # Compile needed version of the program.
    if [[ $block_size == "0" ]]; then
        compile serial
    else
        compile parallel
    fi

    for size in ${sizes[@]}; do
        # Show current values
        printf "BLOCK: $block_size   SIZE: %'d\n" $size

        # Command line arguments to pass to the C program.
        exec_args=($size $block_size)

        # Add leading zeros to the block_size variable in order to create
        # files which can be correctly sorted.
        lzblock_size=$(add_leading_zeros $block_size ${block_sizes[-1]})

        # Subdirectory in which to store measures carried out with current
        # parameters.
        curr_output_dir="$output_dir"/size_$size
        mkdir -p "$curr_output_dir"

        # File in which to store current output.
        output_file="$curr_output_dir"/B${lzblock_size}_S${size}.csv

        measure_time
    done
done

echo "All measures completed."

[[ -z $no_table ]] && python3 "$project_dir"/scripts/evaluate.py "$output_dir"

echo "Done."

safe_exit 0
