# File:   makefile
# Brief:  Makefile used to automate the process of compiling the source files.
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

BIN_DIR := bin
BUILD_DIR := build
INCLUDE_DIR := include
SRC_DIR := src

CC := gcc
NVCC := nvcc
CFLAGS = -g -I $(INCLUDE_DIR)/


.PHONY: all parallel serial dirs clean


all: CFLAGS += --optimize 3
# change this based on the GPU's compute capability.
all: CFLAGS += -arch=sm_75
# remove these if you don't want OpenMP parallelization.
all: CFLAGS += -Xcompiler -fopenmp
all: CLIBS = -lgomp
all: dirs
	$(NVCC) $(CFLAGS) -x cu -dc -c $(SRC_DIR)/util.c $(CLIBS) -o $(BUILD_DIR)/util.o
	$(NVCC) $(CFLAGS) -x cu -dc -c $(SRC_DIR)/main.cu $(CLIBS) -o $(BUILD_DIR)/main.o
	$(NVCC) $(CFLAGS) $(BUILD_DIR)/*.o $(CLIBS) -o $(BIN_DIR)/main.out


parallel: all


serial: CFLAGS += -D_SERIAL -O3
# remove this if you don't want OpenMP parallelization.
serial: CLIBS = -fopenmp
serial: dirs
	$(CC) $(CFLAGS) -c $(SRC_DIR)/util.c $(CLIBS) -o $(BUILD_DIR)/util.o
	$(CC) $(CFLAGS) -c $(SRC_DIR)/serial.c $(CLIBS) -o $(BUILD_DIR)/serial.o
	$(CC) $(CFLAGS) $(BUILD_DIR)/*.o $(CLIBS) -o $(BIN_DIR)/main.out


# Create needed directories if they do not already exist.
dirs:
	$(shell if [ ! -d $(BIN_DIR) ]; then mkdir -p $(BIN_DIR); fi)
	$(shell if [ ! -d $(BUILD_DIR) ]; then mkdir -p $(BUILD_DIR); fi)


# Delete object files and executables.
clean:
	-rm $(BIN_DIR)/* $(BUILD_DIR)/*
