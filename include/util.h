/**
 * @file util.h
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

#ifndef UTIL_H
#define UTIL_H

#include <stdbool.h>
#include <sys/time.h>

/** @brief Minimum integer value accepted in the array. */
#define RANGE_MIN 0

/** @brief Maximum integer value accepted in the array. */
#define RANGE_MAX 255

/**
 * @brief Start measuring the passing of time.
 * @param var: Name of the `double` variable in which to save measurements.
 *
 * The measurement is taken on Wall-Clock time by calling the `gettimeofday()`
 * function.
 */
#define START_TIME(var)                    \
    struct timeval begin_##var, end_##var; \
    gettimeofday(&begin_##var, 0);

/**
 * @brief Stop measuring the passing of time and save the result in `var`.
 * @param var: Name of the `double` variable in which to save measurements.
 *
 * The measurement is taken on Wall-Clock time by calling the `gettimeofday()`
 * function. The function is supposed to have already been called before, via
 * the START_TIME macro, on the same variable.
 * `var` will contain the difference (in seconds) of the values got by the two
 * calls to `gettimeofday()`.
 */
#define END_TIME(var)                                                  \
    gettimeofday(&end_##var, 0);                                       \
    long seconds_##var = end_##var.tv_sec - begin_##var.tv_sec;        \
    long microseconds_##var = end_##var.tv_usec - begin_##var.tv_usec; \
    var = seconds_##var + microseconds_##var * 1e-6;



#ifndef _SERIAL

/**
 * @brief Function that checks whether the last CUDA function call returned with
 *        an error.
 *
 * If an error occurred during the last call to a CUDA API function, the error
 * message is printed out and the program is exited immediately.
 * @param x: CUDA function call.
 */
#define CUDA_ERROR_CHECK(x) {                                    \
    cudaError_t err = x;                                         \
    if (cudaSuccess != err) {                                    \
        fprintf(stderr, "\nERROR CUDA: %s in file %s line %d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__);    \
        exit(EXIT_FAILURE);                                      \
    }                                                            \
}

/**
 * @brief Start counting the passing of time.
 *
 * Create 2 CUDA events: one to signal the start and one the end of the section
 * of code which execution has to be timed.
 * Record the first event.
 * @param var: `float` variable in which to store time result.
 */
#define START_CUDA_TIME(var)                              \
    cudaEvent_t start_##var, stop_##var;                  \
    CUDA_ERROR_CHECK ( cudaEventCreate(&start_##var) );   \
    CUDA_ERROR_CHECK ( cudaEventCreate(&stop_##var) );    \
    CUDA_ERROR_CHECK ( cudaEventRecord(start_##var, 0) );

/**
 * @brief Stop counting the passing of time.
 *
 * Record an event signalling the end of the section of code which execution has
 * to be timed.
 * Save the elapsed time in a float variable, convert it to seconds and destroy
 * the events created to execute this operation.
 * @param var: `float` variable in which to store time result.
 */
#define END_CUDA_TIME(var)                                                    \
    CUDA_ERROR_CHECK ( cudaEventRecord(stop_##var, 0) );                      \
    CUDA_ERROR_CHECK ( cudaEventSynchronize(stop_##var) );                    \
    CUDA_ERROR_CHECK ( cudaEventElapsedTime(&var, start_##var, stop_##var) ); \
    var /= 1000.f;                                                            \
    CUDA_ERROR_CHECK ( cudaEventDestroy(start_##var) );                       \
    CUDA_ERROR_CHECK ( cudaEventDestroy(stop_##var) );

/**
 * @brief Function that checks whether the last Kernel call returned an error.
 *
 * In case an error occurred, the memory is deallocated and the program is
 * exited at once.
 */
#define CUDA_KERNEL_ERROR_CHECK {                   \
    cudaError_t err = cudaGetLastError();           \
    if (err != cudaSuccess) {                       \
        fprintf(stderr, "KERNEL ERROR: %s\n",       \
                cudaGetErrorString(err));           \
        CUDA_ERROR_CHECK( cudaFree(array_device) ); \
        CUDA_ERROR_CHECK( cudaFree(count_device) ); \
        free(array);                                \
        free(count);                                \
        exit(EXIT_FAILURE);                         \
    }                                               \
}

#endif /* ! _SERIAL */



/**
 * @brief Allocate `size` bytes of memory and check that the operation is
 *        successful.
 * @param size: Number of bytes to allocate.
 * @return Pointer to the memory allocated.
 */
void *safe_alloc(long long size);

/**
 * @brief Fill the given array with random integers.
 * @param array: The array.
 * @param size:  Number of elements to generate.
 * @param min:   Minimum value accepted in the array.
 * @param max:   Maximum value accepted in the array.
 */
void array_init_random(int *array, long long size, int min, int max);

/**
 * @brief Find min and max values in the array.
 * @param array: The array.
 * @param size:  Number of elements in the array.
 * @param min:   Minimum value (output).
 * @param max:   Maximum value (output).
 */
void array_min_max(const int *array, long long size, int *min, int *max);

/**
 * @brief Check that the array is sorted in ascending order.
 * @param array: The array.
 * @param size:  Number of elements in the array.
 * @return `true` if the array is sorted; `false` otherwise.
 */
bool array_is_sorted(const int *array, long long size);


#endif /* UTIL_H */
