#!/bin/bash

submit(){
    tar -czf submission.tar.gz lux/ main.py agent.py
    kaggle competitions submit -c lux-ai-season-2 -f submission.tar.gz -m "$1"
}

replay(){
    luxai-s2 main.py main.py -v 2 -s 101 -o replay.html
}

"$@"