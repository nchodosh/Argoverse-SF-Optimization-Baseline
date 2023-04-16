# Arogverse-SF-Optimization-Baseline
An example implementation of Neural Scene Flow Prior for the 2023 Argoverse Self Supervised Scene Flow Competition.

# The Contest
Detailed information about the contest can be found [here](https://eval.ai/web/challenges/challenge-page/2010/submission). But in short, the goal of this contest is to take two succesive LiDAR sweeps taken 0.1s apart and predict the motion that relates them.

# Setting up the dataset

The [user guide](https://argoverse.github.io/user-guide/) has detailed information about how to download the dataset and get the av2 api installed. You will need both to run this method. 

# What is contained in this repository?

This repository is meant to serve as a good starting point for making a test-time optimzation based submission to the contest. We include an implementation of [Neural Scene Flow Prior](https://arxiv.org/abs/2111.01253) along with the improvements proposed in "Re-Evaluating LiDAR Scene Flow"(https://arxiv.org/abs/2304.02150).
