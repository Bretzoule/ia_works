#!/bin/bash

datDocuments=$(ls *.dat)

for doc in $datDocuments; do
    > $doc
done
