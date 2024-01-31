#!/bin/bash

RED='\033[0;31m'
NC='\033[0m'
additional_flags=$1
mergers=("dummy" "frank_wolfe_to_universe" "git_rebasin")

for merger in "${mergers[@]}"; do
    echo -e "${RED}Running experiments for merger $merger ${NC}  "
    python src/scripts/merge_n_models.py "matching/merger=$merger" $additional_flags
    for repair_flag in "True" "False"; do
        python src/scripts/evaluate_merged_model.py "matching/merger=$merger" "matching.repaired=$repair_flag" $additional_flags
    done
    echo -e "${RED}Completed experiments for merger $merger ${NC}  "
done

echo -e "${RED}All experiments completed.${NC}"
