#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Arguments based on Python cell variables
CLUSTER_NAME=$1
PROJECT_ID=$2
LOCATION="us-central1-b" 
CLUSTER_MACHINE_TYPE=$4
CLUSTER_NODE_COUNT=$5
WORKLOAD_POOL=$6
NODEPOOL_NAME=$7
NODEPOOL_MACHINE_TYPE=$8
ACCELERATOR_TYPE=$9
ACCELERATOR_COUNT=${10}
NODEPOOL_NUM_NODES=${11}
SCOPES_RAW=${12}
REPOSITORY_NAME=${13}

# --- GKE Cluster Creation ---
echo "Checking for existing GKE cluster: ${CLUSTER_NAME} in ${LOCATION}"
CLUSTER_EXISTS=$(gcloud container clusters list --filter="name=${CLUSTER_NAME} AND location=${LOCATION}" --format="value(name)" --project="${PROJECT_ID}" 2>/dev/null || true)

if [ -z "${CLUSTER_EXISTS}" ]; then
  echo "GKE cluster ${CLUSTER_NAME} does not exist. Creating zonal cluster..."
  # Use --zone or set location to a specific zone for a zonal cluster
  gcloud container clusters create "${CLUSTER_NAME}" \
    --project "${PROJECT_ID}" \
    --zone "${LOCATION}" \
    --machine-type "${CLUSTER_MACHINE_TYPE}" \
    --num-nodes "${CLUSTER_NODE_COUNT}" \
    --workload-pool "${WORKLOAD_POOL}" \
    --scopes=cloud-platform \
    --addons GcsFuseCsiDriver=ENABLED

  # Fix Workload Identity binding (Corrected variable name and backslash)
  gcloud iam service-accounts add-iam-policy-binding \
    "sdk-training@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/iam.workloadIdentityUser" \
    --member="serviceAccount:${WORKLOAD_POOL}[default/gemma-training-job-ksa]"

  echo "GKE cluster ${CLUSTER_NAME} created successfully."
else
  echo "GKE cluster ${CLUSTER_NAME} already exists."
fi

# Set context for a ZONAL cluster
gcloud container clusters get-credentials "${CLUSTER_NAME}" --zone "${LOCATION}" --project="${PROJECT_ID}"

# --- Node Pool Creation ---
if [ -n "${ACCELERATOR_TYPE}" ] && [ "${ACCELERATOR_COUNT}" -gt 0 ]; then
  echo "Checking for node pool: ${NODEPOOL_NAME}"
  NODEPOOL_EXISTS=$(gcloud container node-pools list --cluster="${CLUSTER_NAME}" --zone="${LOCATION}" --filter="name=${NODEPOOL_NAME}" --format="value(name)" --project="${PROJECT_ID}" 2>/dev/null || true)

  if [ -z "${NODEPOOL_EXISTS}" ]; then
    echo "Creating GPU node pool in ${LOCATION}..."

    gcloud container node-pools create "${NODEPOOL_NAME}" \
      --cluster="${CLUSTER_NAME}" \
      --project="${PROJECT_ID}" \
      --zone="${LOCATION}" \
      --machine-type="${NODEPOOL_MACHINE_TYPE}" \
      --accelerator="type=${ACCELERATOR_TYPE},count=${ACCELERATOR_COUNT}" \
      --num-nodes="${NODEPOOL_NUM_NODES}" \
      --disk-size=200 \
      --enable-autoscaling --min-nodes=0 --max-nodes="${NODEPOOL_NUM_NODES}"
  else
    echo "Node pool already exists."
  fi
fi

# --- Artifact Registry ---
if gcloud artifacts repositories describe "${REPOSITORY_NAME}" --location="us-central1" --project="${PROJECT_ID}" &>/dev/null; then
  echo "Repository exists."
else
  gcloud artifacts repositories create "${REPOSITORY_NAME}" \
    --project="${PROJECT_ID}" \
    --repository-format="docker" \
    --location="us-central1" \
    --description="Docker repository for AI agent images"
fi

echo "Infrastructure setup complete."