#!/bin/bash

submit(){
    tar -cvzf submission.tar.gz main.py agent.py utils.py lux/ models/ wrappers/ best_model
    kaggle competitions submit -c lux-ai-season-2 -f submission.tar.gz -m "$1"
}

pack(){
    tar -cvzf submission.tar.gz main.py agent.py utils.py train.py lux/ models/ wrappers/ best_model latest_model checkpoint
}


replay(){
    luxai-s2 main.py main.py -v 2 -o replay.html
}

train(){
    python train.py --total-timesteps 2000000 --eval-freq 10000 --n-envs 1 --seed 42
}

"$@"