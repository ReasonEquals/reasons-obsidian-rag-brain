---
title: Homelab Setup
status: complete
tags: [infrastructure, self-hosted, proxmox, networking]
started: 2023-06-01
completed: 2023-11-15
summary: Self-hosted infrastructure running Proxmox, Jellyfin, and local AI workloads.
---

# Homelab Setup

A self-hosted infrastructure project: one beefy server running Proxmox as a hypervisor, with VMs for media serving, development tools, and eventually local AI model inference. Built to learn infrastructure concepts hands-on before studying for cloud certs.

## What I built

The main host runs Proxmox VE with the following VMs:

- **Media server** — Jellyfin for local streaming, with storage mounted from an external array
- **Dev tools** — self-hosted Gitea, a local Docker registry, and Portainer for container management
- **AI workloads** — Ollama running local models (Llama 3, Mistral) for offline inference

Networking is handled by pfSense running on a separate mini PC — firewalling between VLANs for the media server (untrusted devices), dev tools (trusted), and AI workloads (air-gapped from WAN for the local models).

## Hardware

Primary host: repurposed workstation with 64GB RAM, 12-core CPU, 2TB NVMe for OS + VMs. The RAM ceiling was the main constraint — Proxmox + all VMs + Ollama running a 13B model saturates it.

Secondary: mini PC running pfSense. Cheap, silent, low power draw (~10W idle).

Storage: 4TB external array on USB3, mounted to the Jellyfin VM via a passthrough. Good enough for a learning project; would use a proper NAS for anything that needed reliability.

## Lessons learned

**Proxmox is excellent for learning virtualization.** The web UI is solid, the documentation is good, and it's close enough to production VMware/KVM patterns that the concepts transfer. Starting here made cloud VM management feel familiar immediately.

**Network segmentation matters even at home.** Running media devices (TVs, Chromecasts) on the same network as development tools was asking for trouble. Once I added VLANs, troubleshooting became dramatically simpler — you know exactly what can talk to what.

**Local AI inference is slow without a GPU.** Ollama on CPU for a 13B model is educational but not practical for daily use. The bottleneck was clear: embeddings and inference both want dedicated hardware. Voyage AI's API is a much better choice for the RAG system than running local embeddings on this hardware.

**Backup discipline is a skill.** Lost a week of Gitea data when a NVMe developed bad sectors and I hadn't verified the backup job was actually running. Always test restores, not just backups.
