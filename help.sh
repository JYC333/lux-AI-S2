#!/bin/bash

submit(){
    tar -cvzf submission.tar.gz main.py agent.py lux/ models/ppo.py utils.py wrappers/ best_model
    kaggle competitions submit -c lux-ai-season-2 -f submission.tar.gz -m "$1"
}

replay(){
    luxai-s2 main.py main.py -v 2 -o replay.html
}

train(){
    python train.py --total-timesteps 2000000 --n-envs 10 --seed 42
}

"$@"