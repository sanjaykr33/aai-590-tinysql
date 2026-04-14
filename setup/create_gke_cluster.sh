#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Arguments based on Python cell variables
CLUSTER_NAME=$1
PROJECT_ID=$2
LOCATION=$3 # Region for the cluster
CLUSTER_MACHINE_TYPE=$4
CLUSTER_NODE_COUNT=$5
WORKLOAD_POOL=$6
NODEPOOL_NAME=$7
NODEPOOL_MACHINE_TYPE=$8
ACCELERATOR_TYPE=$9
ACCELERATOR_COUNT=${10}
NODEPOOL_NUM_NODES=${11}
SCOPES_RAW=${12} # Comma-separated scopes
REPOSITORY_NAME=${13}
ZONE=${15}
SCOPES_ARG=SCOPES_RAW

# --- GKE Cluster Creation ---
echo "Checking for existing GKE cluster: ${CLUSTER_NAME} in ${ZONE}"
CLUSTER_EXISTS=$(gcloud container clusters list --filter="name=${CLUSTER_NAME} AND location=${ZONE}" --format="value(name)" --project="${PROJECT_ID}" 2>/dev/null || true)

if [ -z "${CLUSTER_EXISTS}" ]; then
  echo "GKE cluster ${CLUSTER_NAME} does not exist. Creating..."
  gcloud container clusters create "${CLUSTER_NAME}" \
    --project "${PROJECT_ID}" \
    --location "${ZONE}" \
    --machine-type "${CLUSTER_MACHINE_TYPE}" \
    --num-nodes "${CLUSTER_NODE_COUNT}" \
    --workload-pool "${WORKLOAD_POOL}" \
    --scope=cloud-platform

  gcloud container clusters update "${CLUSTER_NAME}" \
    --update-addons GcsFuseCsiDriver=ENABLED \
    --region "${ZONE}"

  gcloud iam service-accounts add-iam-policy-binding \
    ${GCP_SERVICE_ACCOUNT}
    --role="roles/iam.workloadIdentityUser" \
    --member="serviceAccount:${WORKLOAD_POOL}[default/default]"


  if [ $? -eq 0 ]; then
    echo "GKE cluster ${CLUSTER_NAME} created successfully."
  else
    echo "Failed to create GKE cluster ${CLUSTER_NAME}. Exiting."
    exit 1
  fi
else
  echo "GKE cluster ${CLUSTER_NAME} already exists. Skipping cluster creation."
fi

# Ensure kubectl context is set for the cluster
echo "Setting kubectl context for cluster: ${CLUSTER_NAME}"
gcloud container clusters get-credentials "${CLUSTER_NAME}" --region "${ZONE}" --project "${PROJECT_ID}"

# --- Node Pool Creation (if accelerator type is specified) ---
if [ -n "${ACCELERATOR_TYPE}" ] && [ "${ACCELERATOR_COUNT}" -gt 0 ]; then
  echo "Checking for existing node pool: ${NODEPOOL_NAME} in cluster ${CLUSTER_NAME}"
  NODEPOOL_EXISTS=$(gcloud container node-pools list --cluster="${CLUSTER_NAME}" --region="${ZONE}" --filter="name=${NODEPOOL_NAME}" --format="value(name)" --project="${PROJECT_ID}" 2>/dev/null || true)

  if [ -z "${NODEPOOL_EXISTS}" ]; then
    echo "Node pool ${NODEPOOL_NAME} does not exist. Creating..."
    gcloud container node-pools create "${NODEPOOL_NAME}" \
      --cluster="${CLUSTER_NAME}" \
      --project="${PROJECT_ID}" \
      --region="${ZONE}" \
      --machine-type="${NODEPOOL_MACHINE_TYPE}" \
      --accelerator="type=${ACCELERATOR_TYPE},count=${ACCELERATOR_COUNT}" \
      --num-nodes="${NODEPOOL_NUM_NODES}" \
      --enable-autoscaling --min-nodes="0" --max-nodes="${NODEPOOL_NUM_NODES}"

    if [ $? -eq 0 ]; then
      echo "Node pool ${NODEPOOL_NAME} created successfully."
    else
      echo "Failed to create node pool ${NODEPOOL_NAME}. Exiting."
      exit 1
    fi
  else
    echo "Node pool ${NODEPOOL_NAME} already exists. Skipping creation."
  fi
else
  echo "No accelerator type specified or accelerator count is 0. Skipping dedicated node pool creation."
fi

# --- Artifact Registry Docker Repository Creation ---
echo "Checking for existing Artifact Registry repository: ${REPOSITORY_NAME} in ${LOCATION})"
# Use gcloud artifacts repositories describe and check its exit code
if gcloud artifacts repositories describe "${REPOSITORY_NAME}" --location="${LOCATION}" --project="${PROJECT_ID}" &>/dev/null; then
  echo "Artifact Registry repository ${REPOSITORY_NAME} already exists. Skipping creation."
else
  echo "Artifact Registry repository ${REPOSITORY_NAME} does not exist. Creating..."
  gcloud artifacts repositories create "${REPOSITORY_NAME}" \
    --project="${PROJECT_ID}" \
    --repository-format="docker" \
    --location="${LOCATION}" \
    --description="Docker repository for AI agent images"

  if [ $? -eq 0 ]; then
    echo "Artifact Registry repository ${REPOSITORY_NAME} created successfully."
  else
    echo "Failed to create Artifact Registry repository ${REPOSITORY_NAME}. Exiting."
    exit 1
  fi
fi

echo "Infrastructure setup complete."
