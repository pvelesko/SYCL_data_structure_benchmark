#!/bin/bash

export OMP_NUM_THREADS=4
SIZE=25000 # 1MB

for FILL in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
  RESDIR="./profiles/amplxe_${SIZE}_${FILL}"
  CMD="./intel ${SIZE} 10000 ${FILL}"
  amplxe-cl -c hpc-performance --result-dir $RESDIR -- $CMD
done

