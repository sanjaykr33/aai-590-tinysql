#!/bin/bash

# Check if kubectl is already installed
if ! command -v kubectl &> /dev/null
then
    echo "kubectl not found. Installing..."

    # Download the latest stable release of kubectl
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"

    # Make the kubectl binary executable
    chmod +x ./kubectl

    # Move the binary to a directory in your PATH (e.g., /usr/local/bin)
    sudo mv ./kubectl /usr/local/bin/kubectl

    # Verify the installation
    kubectl version --client

    echo 'kubectl installed successfully!'
else
    echo 'kubectl is already installed.'
    kubectl version --client
fi