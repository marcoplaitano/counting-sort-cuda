/**
 * @file util.c
 * @brief This file contains some useful, general-purpose functions.
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

#include "util.h"

#ifdef _OPENMP
    #include <omp.h>
#else
    #define omp_get_thread_num() 0
#endif
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


void *safe_alloc(long long size) {
    if (size < 1) {
        fprintf(stderr, "Can not allocate memory of %lld bytes.\n", size);
        exit(EXIT_FAILURE);
    }

    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Could not allocate memory of %lld bytes.\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}


void array_init_random(int *array, long long size, int min, int max) {
    long long i = 0;

    #pragma omp parallel num_threads(8) shared(array, size) private(i)
    {
        unsigned seed = time(NULL) ^ omp_get_thread_num();
        #pragma omp for
        for (i = 0; i < size; i++)
            array[i] = rand_r(&seed) % (max + 1 - min) + min;
    }
}


void array_min_max(const int *array, long long size, int *min, int *max) {
    *min = array[0];
    *max = array[0];
    long long i = 0;

    #pragma omp parallel for num_threads(8) default(shared) private(i)
    for (i = 0; i < size; i++) {
        if (array[i] < *min) {
            #pragma omp critical
            {
                if (array[i] < *min)
                    *min = array[i];
            }
        }
        else if (array[i] > *max) {
            #pragma omp critical
            {
                if (array[i] > *max)
                    *max = array[i];
            }
        }
    }
}


bool array_is_sorted(const int *array, long long size) {
    long long i = 0;
    bool flag = true;

    #pragma omp parallel for num_threads(8) private(i) shared(array, size, flag)
    for (i = size - 1; i > 0; i--) {
        if (array[i] < array[i - 1]) {
            #pragma omp critical
            flag = false;
        }
    }

    return flag;
}
