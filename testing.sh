#!/bin/bash

# loop over all the png
END=5
for i in $(seq 1 $END); do
    python3 main.py --Path "Handwritten/$i.png"; 
done