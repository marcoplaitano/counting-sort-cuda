/**
 * @file serial.c
 * @brief Main program performing a serial version of the Counting Sort
 *        algorithm.
 * @author Marco Plaitano
 * @date 19 Jan 2022
 *
 * COUNTING SORT CUDA
 * Parallelize and Evaluate Performances of "Counting Sort" Algorithm, by using
 * CUDA.
 *
 * Copyright (C) 2022 Plaitano Marco
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "util.h"


/**
 * @brief Sort the given array using Counting Sort.
 * @param array: The array.
 * @param size:  Number of elements in the array.
 *
 * The sorting happens in place.
 */
void counting_sort(int *array, long long size);



int main(int argc, char **argv) {
    /* Check for the correct amount of command line arguments. */
    if (argc < 2) {
        fprintf(stderr, "ERROR! usage: bin/serial.out array_size\n");
        exit(EXIT_FAILURE);
    }

    /* Check for the correctness of the parameters. */
    if (RANGE_MAX <= RANGE_MIN) {
        fprintf(stderr, "ERROR! can't have RANGE_MAX <= RANGE_MIN.\n");
        exit(EXIT_FAILURE);
    }

    float time_sort = 0;
    long long size = atoll(argv[1]);
    int *array = (int *)safe_alloc(size * sizeof(int));
    array_init_random(array, size, RANGE_MIN, RANGE_MAX);

    START_TIME(time_sort);
    counting_sort(array, size);
    END_TIME(time_sort);

    /* Test correctness of the algorithm. */
    bool is_sorted = array_is_sorted(array, size);
    free(array);

    /* Show output and exit. */
    if (!is_sorted) {
        fprintf(stderr, "Array NOT sorted!!!\n");
        return EXIT_FAILURE;
    } else {
        printf("%lld;0;0;0;%.5f\n", size, time_sort);
        return EXIT_SUCCESS;
    }
}




void counting_sort(int *array, long long size) {
    long long i = 0, j = 0, k = 0;
    int min = 0, max = 0;
    array_min_max(array, size, &min, &max);
    const int count_size = max - min + 1;

    int *count = (int *)safe_alloc(sizeof(int) * count_size);
    for (i = 0; i < count_size; i++)
        count[i] = 0;

    for (i = 0; i < size; i++)
        count[array[i] - min] += 1;

    for (i = min; i < max + 1; i++)
        for (j = 0; j < count[i - min]; j++)
            array[k++] = i;

    free(count);
}
