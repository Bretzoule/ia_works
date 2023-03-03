#!/bin/bash

set -x

gnuplot -p -e "plot 'neuralNetIterations.dat' using 1:2 with lines"
gnuplot -p -e "plot 'errorWithNoiseForOne.dat' with lines title 'One', 'errorWithNoiseForZero.dat' with lines title 'Zero', 'errorWithNoiseForTwo.dat' with lines title 'Two', 'errorWithNoiseForThree.dat' with lines title 'Three', 'errorWithNoiseForFour.dat' with lines title 'Four', 'errorWithNoiseForFive.dat' with lines title 'Five', 'errorWithNoiseForSix.dat' with lines title 'Six', 'errorWithNoiseForSeven.dat' with lines title 'Seven', 'errorWithNoiseForEight.dat' with lines title 'Eight', 'errorWithNoiseForNine.dat' with lines title 'Nine'"
