# Udacity Deep Reinforcement Learning Nanodegree Project: Reacher

## Project Details

The objective of this project is to solve the Unity ML-Agents task, `Reacher`, in which continuous 
control of 2-link robot arms are maintained in the attempt to keep the robot arms within a 
moving goal volume.  A video of the environment is available [here](https://www.youtube.com/watch?v=2N9EoF6pQyE&feature=youtu.be).

### Specifics of the Task
* The state space is made of 33 observations of continuous features that exist in the range of [-1, 1].
* The action space consists of 4 continuous actions representing the torques applied to the 2 joints.  The values for 
each feature of the action must be within [-1, 1]. 
* This version of the environment runs 20 agents in parallel.  
* Agents are rewarded with a score of 0.1 for each step in which the end of the robot arm is 
within the floating goal volume, and a score of 0.0 for each step in which the robot arm does not
end within the floating goal volume.  
* The environment is considered solved when all 20 agents obtain an average episode score of +30.0 over the course 
of 100 episodes.   

## Installation

This repository runs within the provided docker environment. The base image upon which this 
repository's docker image is built is freely available from my DockerHub, 
`ccthompson82/drlnd:0.0.7`.  No downloads are necessary if the instructions below are followed. 

### Dependencies
* Python 2.7 or Python 3.5
* Docker version 17 or later
    - [docker](https://docs.docker.com/install/)
    
## Setup the docker image

1. Update the data directory in the Makefile of this project's repository.  
    * Modify the environment variable definition on line 37.  It can be advantageous to mount a storage directory,
     though a default option would simply be `DATA_SOURCE=$(PWD)/data` to keep and track data locally in this 
     repository.    
     
2. Setup the development environment in a Docker container with the following command:
    - `make init`
    
    This command gets the resources for training and testing, and then prepares the Docker image for the experiments.
    
## Launching the docker container

1. After creating the Docker image, run the following command.

- `make create-container`

    The above command creates a Docker container from the Docker image which we create with `make init`, and then
login to the Docker container.  This command needs to be run only once after creating the docker image.  After the
container hs been created with the command above, use the following command to enter the existing container: `make start-container`.

# Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [cookiecutter-docker-science](https://docker-science.github.io/) project template.
