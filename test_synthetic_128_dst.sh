#!/bin/bash

python3 test.py \
    -d synthetic \
    -wbs D S T \
    -mn ifrnet_p_128_D_S_T \
    -ted ../mixedillWB/data/synthetic-{} \
    -od ./results/ifrnet/images/synthetic/ \
    -g 1
