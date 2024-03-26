#!/bin/bash

RED='\033[0;31m'
NC='\033[0m'

model_pairs=("1 2" "2 3" "1 3")

for approach in "frank_wolfe" "git_rebasin" "dummy"
do
    for pair in "1,2" # "2,3" "1,3"
    do
        for seed in 1 2 3 4
        do
            echo -e "${RED}Running experiments for model pair $pair with seed $seed${NC}  "

            python src/scripts/match_two_models.py "matching/matcher=$approach" "matching.model_seeds=[$pair]" "matching.seed_index=$seed"
            python src/scripts/evaluate_matched_models.py "matching/matcher=$approach" "matching.model_seeds=[$pair]" "matching.seed_index=$seed"

            echo -e "${RED}Completed experiments for model pair $pair with seed $seed${NC}"
        done
    done
done

echo -e "${RED}All experiments completed.${NC}"
