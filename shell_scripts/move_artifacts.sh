#!/bin/bash
ckpts=("ResNet22_4_1" "ResNet22_4_2" "ResNet22_4_3" "ResNet22_4_4" "ResNet22_4_5" "VGG16_1" "VGG16_2" "VGG16_3" "VGG16_4" "VGG16_5" "ResNet22_8_1" "ResNet22_8_2" "ResNet22_8_3" "ResNet22_16_1" "ResNet22_16_2" "ResNet22_16_3" "ResNet22_2_1" "ResNet22_2_2" "ResNet22_2_3" "ResNet22_2_4" "ResNet22_2_5")

for ckpt in "${ckpts[@]}"
do
    wandb artifact get crisostomi/cycle-consistent-model-merging/${ckpt}:v0
    # get the content of
    content=$(ls artifacts/${ckpt}:v0/)
    wandb artifact put --name gladia/cycle-consistent-model-merging/${ckpt} artifacts/${ckpt}:v0/${content} --type 'checkpoint'
done
