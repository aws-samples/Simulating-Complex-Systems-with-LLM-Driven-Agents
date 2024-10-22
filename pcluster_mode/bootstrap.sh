#!/bin/bash
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                #
######################################################################

# Update and install system dependencies
sudo apt-get update
sudo apt-get install software-properties-common curl -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt install python3.11 python3.11-distutils python3.11-dev -y

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3.11 get-pip.py

# Install Ray
sudo python3.11 -m pip install ray langchain_ollama numpy
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
