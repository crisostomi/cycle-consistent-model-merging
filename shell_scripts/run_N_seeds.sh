#!/bin/bash

RED='\033[0;31m'
NC='\033[0m'

for seed in 31 32 33 34 35 36 37 38 39 #11 12 13 14 15 16 17 18 19 20 #1 2 3 4 5 6 7 8 9 10
do
    echo -e "${RED}Running experiments for seed $seed ${NC}  "

    python src/scripts/match_two_models.py "matching.seed_index=$seed"
    python src/scripts/evaluate_matched_models.py "matching.seed_index=$seed"

    echo -e "${RED}Completed experiments for seed $seed ${NC}"
done

echo -e "${RED}All experiments completed.${NC}"
