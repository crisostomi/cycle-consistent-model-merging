#!/bin/bash

RED='\033[0;31m'
NC='\033[0m'

model_pairs=("1 2" "2 3" "1 3")

for pair in "1,2" "2,3" "1,3"
do
    echo -e "${RED}Running experiments for model pair $pair ${NC}  "

    python src/scripts/match_two_models.py "matching.model_seeds=[$pair]"
    python src/scripts/evaluate_matched_models.py "matching.model_seeds=[$pair]"

    echo -e "${RED}Completed experiments for model pair $pair ${NC}"
done

echo -e "${RED}All experiments completed.${NC}"
