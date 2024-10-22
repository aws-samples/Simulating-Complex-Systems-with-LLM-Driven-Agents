# Simulating Complex Systems with LLM-Driven Agents

## Overview

This repository contains code for an advanced Agent-Based Model (ABM) simulation of energy supply chains, incorporating Large Language Models (LLMs) to represent complex decision-making processes of various agents, including energy producers and utilities. The simulation aims to capture nuanced behaviors, including emotional responses and intricate interactions, going beyond traditional mathematical rule-based ABMs. This is NOT a validated model and serves as an example for someone to learn from. Please see the [blog](www.google.com) which discuss this repo in more details. 

## Key Features

- Integration of LLMs for agent decision-making
- Scalable architecture using AWS ParallelCluster and Ray
- Ollama for fast LLM inference
- Customizable agent personas (e.g., environmentally conscious, greedy, depressed)
- Analysis of market dynamics, including price competition and regulation effects

## Requirements

- Python 3.x
- AWS account with appropriate permissions
- [Ollama](https://ollama.com/)
- [Ray](https://www.ray.io/)
- Additional Python packages (specified in `requirements.txt`)

## Setup

1. Clone this repository
2. Set up an [AWS ParallelCluster](https://docs.aws.amazon.com/parallelcluster/latest/ug/pcui-using-v3.html).
    An example configuration can be found in: ```parallelCluster_config.yaml```. Users are encouraged to familiarize with how ParallelCluster is setup and modify for the size of the ABM of interest. Our example leverages a g5.16xlarge EC2 instance to ensure Ollama efficiently uses a single GPU. 

    **Notes:** 
    - The scripts in this project use package management commands (e.g., `apt-get`) specific to Debian-based Linux distributions. For full compatibility, use Ubuntu or another Debian-based OS for your cluster nodes.
    - Ensure your ParallelCluster configuration includes an associated EFS (Elastic File System) volume. EFS is used to faciliate parallel installations across nodes.
3. Once the ParallelCluster fleet has been provisioned we need to install python dependancies and Ollama.  Login (e.g. SSH) to the ParallelCluster head node and run the bootstrap.sh to update the OS, install python software, and install Ollama:
```./pcluster_mode/bootstrap.sh```
In addition, the head node will also require the installation of the python dependancy list:

    ```python3.11 -m pip install -r requirments.txt```

4. Next we will run the bootstrap.sh on each compute node, it can optionally be implemented as a start up script inside the ParalleCluster depooyment. 

    ```python ./pcluster_mode/update_compute_nodes.py --action bootstrap```

5. Next start ray on the head node: ```ray start --head```, carefully note the IP address that is displayed on the screen, which will look like ```999.99.999.99:1234```

    **Note on Python Versions:**
    - The bootstrap script installs Python 3.11 on the compute nodes. To ensure compatibility and avoid version mismatch errors when starting Ray, make sure you're using Python 3.11 on the head node as well. If you encounter an error stating that Ray was started with a different Python version, verify that both the head node and compute nodes are using Python 3.11. This should not be an issue if you use the bootstrap script on the head node, as per Step 2.
6. Then start Ray on each compute node 

    ```python ./pcluster_mode/update_compute_nodes.py --action ray-start --ray-address <RAY ADDRESS>```

7. Now we should run ```ray status``` to see if all resources have been cataloged by Ray.  Hence, if you provisioned 5 GPUs with ParallelCluster, then Ray should have 5 GPUs in its resource pool. 
8. Last we need to download the LLM weights to each compute node and the head node.  We are using Llama 3.x and thus the command is: 

    ```python ./pcluster_mode/update_compute_nodes.py --action ollama --model "llama3.2:3b-instruct-fp16"```

## Usage

1. Run the main simulation script: `python energyABM_LLM_ollama.py`
    - To change the number of companies or number of timesteps in the simulation, modify the call to the function `run_simulation()` at the end of the script:
    ```python
    #%% main
    if __name__ == "__main__":
        run_simulation(num_companies=20, num_steps=50)    
        print("Simulation completed. Check the generated PKL files for saved data.")
    ```

    In this example, `run_simulation()` is called with 20 companies over 50 time steps.
    - The simulations should output a series of .pkl files containing the results of the simulations (i.e. simulated data).
2. Analyze results using provided streamlit app. Notice that streamlit creates a server on the head node.  You will need to use port forwarding to enable viewing from your local browser. One possible solution is through the use of [VisualStudioCode](https://code.visualstudio.com/docs/editor/port-forwarding).

```streamlit run streamlit_energy_subs.py```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

