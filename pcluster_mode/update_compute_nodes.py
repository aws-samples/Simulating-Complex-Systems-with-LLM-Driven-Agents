#!/usr/bin/env python3
# -*- coding: utf-8 -*-
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                #
######################################################################

import subprocess
import sys
from joblib import Parallel, delayed
import argparse
import os


# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Path to your bootstrap script
BOOTSTRAP_SCRIPT = os.path.join(current_dir, "bootstrap.sh")

# Path to your rm_models script
RM_MODELS_SCRIPT = os.path.join(current_dir, "rm_ollama_models.sh")


def get_slurm_nodes():
    """Get a list of all nodes in the SLURM cluster"""
    try:
        result = subprocess.run(["sinfo", "-h", "-o", "%n"], capture_output=True, text=True, check=True)
        return result.stdout.strip().split('\n')
    except subprocess.CalledProcessError as e:
        print(f"Error getting SLURM nodes: {e}", file=sys.stderr)
        sys.exit(1)

def run_command(node, command):
    """Run a command on a single node"""
    try:
        subprocess.run(["ssh", node, command], check=True)
        return f"Command '{command}' completed successfully on {node}"
    except subprocess.CalledProcessError as e:
        return f"Error running command '{command}' on {node}: {e}"

def run_bootstrap(node):
    """Run bootstrap script on a single node"""
    return run_command(node, f"bash -s < {BOOTSTRAP_SCRIPT}")


def run_rm_models(node):
    """Run rm_models script on a single node"""
    return run_command(node, f"bash -s < {RM_MODELS_SCRIPT}")
#%%

def main():
    parser = argparse.ArgumentParser(description="Run commands on SLURM cluster nodes.")    
    parser.add_argument("--action", choices=["bootstrap", "ollama", "ray-stop", "ray-start", "rm-models"], required=True,
                       help="Action to perform on nodes")
    parser.add_argument("--model", help="Model name for ollama pull")
    parser.add_argument("--ray-address", help="IP address for ray start")
    args = parser.parse_args()

    nodes = get_slurm_nodes()
    nodes = [x for x in nodes if '-dy-' not in x]
    print(f"Found {len(nodes)} nodes in the SLURM cluster")

    if args.action == "bootstrap":
        command_func = run_bootstrap        
    elif args.action == "ollama":
        if not args.model:
            print("Error: --model is required for ollama action", file=sys.stderr)
            sys.exit(1)
        command_func = lambda node: run_command(node, f"ollama pull {args.model}")
    elif args.action == "ray-stop":
        command_func = lambda node: run_command(node, "ray stop")
    elif args.action == "ray-start":
        if not args.ray_address:
            print("Error: --ray-address is required for ray-start action", file=sys.stderr)
            sys.exit(1)
        command_func = lambda node: run_command(node, f"ray start --address={args.ray_address}")
    elif args.action == "rm-models":
        command_func = run_rm_models

    # Run commands in parallel
    results = Parallel(n_jobs=-1)(delayed(command_func)(node) for node in nodes)

    # Print results
    for result in results:
        print(result)

if __name__ == "__main__":
    """
    Usage:
    - For bootstrap: `python script.py --action bootstrap`
    - For ollama pull: `python script.py --action ollama --model <model_name>`
    - For ray stop: `python script.py --action ray-stop`
    - For ray start: `python script.py --action ray-start --ray-address <ip_address>`
    - For removing models: `python script.py --action rm-models`
    """
    main()