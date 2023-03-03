#!/bin/bash

set -x

gnuplot -p -e "plot 'neuralNetIterations.dat' using 1:2 with lines"
gnuplot -p -e "plot 'errorWithNoiseForOne.dat' with lines title 'One', 'errorWithNoiseForZero.dat' with lines title 'Zero'"
