#! /bin/bash

unar epsilon_normalized.bz2
gcc -march=native -mtune=native -O3 extract_train.c -o extract_train
./extract_train

unar epsilon_normalized.t.bz2
gcc -march=native -mtune=native -O3 extract_test.c -o extract_test
./extract_test