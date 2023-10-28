#!/bin/bash

# Remove CMakeCache.txt
rm CMakeCache.txt

# Remove CMakeFiles directory
rm -rf CMakeFiles/

# Run cmake in the current directory
cmake .
