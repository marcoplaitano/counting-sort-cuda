# COUNTING SORT CUDA

This project presents a version of [Counting Sort] Algorithm that has been
parallelized by use of the [CUDA] environment on **NVIDIA GPUs**.

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

## Counting Sort

Counting Sort is an algorithm for sorting a collection of objects according to
their respective keys (small, positive integers).  
The assumption made on the input array is that it must be filled either with
integers in a range **[min, max]** or any other type of elements which can be
represented each with a unique key in that range.  
This algorithm does not compare the elements but rather counts their frequency
in the array to determine their new position.

Below, the pseudocode for the algorithm:

```c
1   initialize array[N]
    // find min and max in the array
2   min <- array[0]
3   max <- array[0]
4   for i <- 1 to N
5       if array[i] < min
6           min <- array[i]
7       if array[i] > max
8           max <- array[i]
9   end for
    // initialize count array with a size equal to the [min, max] range
10  range <- max - min + 1
11  initialize count[range]
    // count the number of occurrences of each element
12  for i <- 0 to N
13      count[array[i] - min] <- count[array[i] - min] + 1
14  end for
    // reposition the elements based on their number of occurrences
15  initialize z <- 0
16  for i <- min to max
17      for j <- 0 to count[i - min]
18          array[z] <- i
19          z <- z + 1
20      end for
21  end for
```

Worst-case Time Complexity: **O(n + k)**  
Where *n* is the number of elements in the array and *k* is the size of the
**[min, max]** range.

Worst-case Space Complexity: **O(n + k)**.  
If the range **[min, max]** is far greater than the number of elements, the
auxiliary array introduces a considerable amount of wasted space.

Another trait of this algorithm is its *Stability*: elements sharing the same
key will retain their relative positions.

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

## Performance Evaluation

To verify and estimate the benefits of parallelization, a [shell cript] can
[be executed] to run the C program multiple times, with different combinations
of these parameters:

+ Array size
+ Block and Grid size

The shell script calls (if not [told otherwise]) a [Python script] to create
tables in order to easily compare the execution times of the different versions.

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

## Usage

In order to be able to execute the following script, *cd* into the project's
root directory and give it executable permissions:

```shell
chmod +x ./scripts/*.sh
```

### Generate the data

To generate the data regarding the execution times evaluation and comparison,
launch the following shell script.  
Take a look at the [dependencies] required to run the script.

```shell
./scripts/measures.sh [OPTION]...
```

#### Options and Parameters for Measures

The script supports the definition of various options and/or parameters via
command line arguments.

| Argument                       | Description               |
| :---                           | :----                     |
| -h, --help                     | Show guide and quit.      |
| -n **N**, --num-measures **N** | Run every measure **N** times. (default is 20)\* |
| -d **DIR**, --dir **DIR**      | Specify output directory. (default is *./output*) |
| --no-table                     | Do not run the Python script to create the tables. |

\* *The higher the number, the more precise the mean value is.*

Other parameters, like the block sizes to use or the array size(s), can be
modified inside the script [itself].

### Compile source files

To compile source files without the help of the shell script, use the makefile.

Compile sources to generate parallel version that runs on **NVIDIA** GPUs:

```shell
make parallel
```

Or compile serial version that runs on any CPU:

```shell
make serial
```

In both cases the executable file produced is *bin/main.out*.

Check and modify the [makefile] based on the GPU's *Compute Capability* and to
toggle the OpenMP linking.

### Clean

To delete object files and executables:

```shell
make clean
```

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

## Dependencies

A list of *HW & SW* requirements to run the program and the scripts:

+ NVIDIA GPU
+ Bash Shell 4.2+
+ nvcc 9+
+ make
+ OpenMP 4.5+
+ Python 3.7+

### Python modules

The following modules are needed to handle CSV files and tables and perform
some calculations:

+ scipy
+ numpy
+ pandas
+ prettytable

Use **pip** to install these with the following command:

```shell
pip install *module_name*
```

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

## Author

Marco Plaitano

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

## License

Distributed under the [GPLv3] license.

<!-- LINKS -->

[Counting Sort]:
https://en.wikipedia.org/wiki/Counting_sort
"Wikipedia article"

[CUDA]:
https://developer.nvidia.com/cuda-toolkit
"Main website"

[shell cript]:
scripts/measures.sh
"Repository file"

[be executed]:
#generate-the-data
"Anchor to header"

[told otherwise]:
#options-and-parameters-for-measures
"Anchor to header"

[Python script]:
scripts/evaluate.py
"Repository file"

[dependencies]:
#dependencies
"Anchor to header"

[itself]:
scripts/measures.sh
"Repository file"

[makefile]:
makefile
"Repository file"

[GPLv3]:
LICENSE
"Repository file"
