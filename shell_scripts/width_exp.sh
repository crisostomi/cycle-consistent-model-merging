#!/bin/bash

RED='\033[0;31m'
NC='\033[0m'

for width in 1 2 4 8 16; do
    echo -e "${RED}Running experiments for width $width ${NC}  "

    python src/scripts/merge_n_models.py "model.widen_factor=$width"
    python src/scripts/evaluate_merged_model.py "matching.repaired=False" "model.widen_factor=$width"
    echo -e "${RED}Completed experiments for width $width ${NC}"
done

echo -e "${RED}All experiments completed.${NC}"
