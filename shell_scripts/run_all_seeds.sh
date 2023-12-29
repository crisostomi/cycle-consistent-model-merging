#!/bin/bash

RED='\033[0;31m'
NC='\033[0m'

for seed in 1 2 3 4
do
    echo -e "${RED}Running experiments with seed $seed${NC}"

    python src/scripts/match_two_models.py "matching.seed_index=$seed"
    python src/scripts/evaluate_matched_models.py "matching.seed_index=$seed"

    echo -e "${RED}Completed experiments for seed $seed${NC}"
done

echo -e "${RED}All experiments completed.${NC}"
