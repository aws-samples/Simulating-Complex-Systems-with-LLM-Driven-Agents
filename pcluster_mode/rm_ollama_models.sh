#!/bin/bash
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                #
######################################################################

# Run 'ollama list' and store the output
output=$(ollama list)

# Skip the header line and process each line of the output
echo "$output" | tail -n +2 | while read -r line
do
    # Extract the NAME field (first field)
    name=$(echo "$line" | awk '{print $1}')
    
    # Run 'ollama rm' for each NAME
    echo "Removing $name..."
    ollama rm "$name"
done

echo "All listed models have been removed."
