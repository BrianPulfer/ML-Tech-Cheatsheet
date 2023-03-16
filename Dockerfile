# syntax=docker/dockerfile:1

# Examples:
#   PyTorch's Dockerfile: https://github.com/pytorch/pytorch/blob/master/Dockerfile

######################## IMAGE BUILDING PART ########################
# Docker image variables (can have default values)
ARG BASE_IMAGE=pytorch/pytorch

# Creating the OS (PyTorch Image)
FROM ${BASE_IMAGE} as base-image

# Working directory
WORKDIR /app

# Copying files
COPY . .

# Creating conda environment within docker
RUN conda env create --name ml_cheatsheet --file environment.yml

######################## IMAGE RUNNING PART ########################
# Activate the conda environment
SHELL ["conda", "run", "-n", "ml_cheatsheet", "/bin/bash", "-c"]

# Checking if CUDA is available, showing conda environments and running pytest
CMD echo -n "Checking CUDA is available: " && python -c 'import torch; print(torch.cuda.is_available())' \
    && conda env list \
    && pytest ./test
