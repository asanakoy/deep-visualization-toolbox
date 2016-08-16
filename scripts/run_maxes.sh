#!/bin/bash -x
./scripts/find_maxes.sh
./scripts/crop_max_patches.sh
./scripts/gen_max_grids.sh