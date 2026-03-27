# DocQA AI Helm Chart

## Introduction

This Helm chart deploys DocQA AI on Kubernetes.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.2.0+
- PV provisioner support in the underlying infrastructure

## Installing the Chart

```bash
# Add the repository (if using a repo)
helm repo add docqa https://charts.docqa-ai.com
helm repo update

# Install the chart
helm install my-docqa docqa/docqa-ai --values values.yaml
