#!/bin/bash

RED='\033[0;31m'
NC='\033[0m'

model_pairs=("1 2" "2 3" "1 3")

for pair in "1,2" "2,3" "1,3"
do
    for approach in "dummy" "frank_wolfe"
    do
        echo -e "${RED}Running experiments for model pair $pair ${NC}"

        python src/scripts/match_two_models.py "matching.model_seeds=[$pair]" "matching/matcher=$approach"
        python src/scripts/evaluate_matched_models.py "matching.model_seeds=[$pair]" "matching/matcher=$approach"

        echo -e "${RED}Completed experiments for model pair $pair ${NC}"
    done
done

for pair in "1,2" "2,3" "1,3"
do
    for seed in 1 2 3 4
    do
        echo -e "${RED}Running experiments for model pair $pair ${NC}"

        python src/scripts/match_two_models.py "matching.model_seeds=[$pair]" "matching/matcher=git_rebasin" "matching.seed_index=$seed"
        python src/scripts/evaluate_matched_models.py "matching.model_seeds=[$pair]" "matching/matcher=git_rebasin" "matching.seed_index=$seed"

        echo -e "${RED}Completed experiments for model pair $pair ${NC}"
    done
done

echo -e "${RED}All experiments completed.${NC}"
