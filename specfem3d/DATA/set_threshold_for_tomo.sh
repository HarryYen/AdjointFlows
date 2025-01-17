#!/bin/bash

awk '{
    if (NR > 4);
    if ($4 < 2600) $4 = 2600;
    if ($5 < 1500) $5 = 1500;
    printf "%13.3f %13.3f %13.3f %10.3f %10.3f %10.3f\n", $1, $2, $3, $4, $5, $6
}' tomography_model.xyz > output.txt
