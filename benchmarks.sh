#!/bin/bash

python train.py\
    train_cycles=25\
    seed=0\
    env.random_goal=true\
    augment.rollouts=false,true\
    augment.HER=false,true\
    augment.IER=false,true\
    --multirun
