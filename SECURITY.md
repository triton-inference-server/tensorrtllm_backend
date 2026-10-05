<!--
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# Security Policy

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
discussions, or pull requests.**

To report a potential security vulnerability in this project or any other
NVIDIA product, use one of the following channels:

1. **NVIDIA Vulnerability Disclosure Program (preferred):**
   [https://www.nvidia.com/en-us/security/](https://www.nvidia.com/en-us/security/)
2. **Email:** [psirt@nvidia.com](mailto:psirt@nvidia.com). Please encrypt
   sensitive reports with NVIDIA's
   [PGP key](https://www.nvidia.com/en-us/security/pgp-key).
3. **GitHub Private Vulnerability Reporting (where enabled):** use the **Security** tab of this
   repository and select **Report a vulnerability**.

**OEM partners should contact their NVIDIA Customer Program Manager.**

Please include as much of the following as you can:

1. Product name and version or branch (for example the container tag or the
   `tensorrtllm_backend` / TensorRT-LLM release) that contains the issue
2. Type of vulnerability (for example code execution, denial of service,
   memory corruption, information disclosure)
3. Step-by-step instructions to reproduce the issue
4. Proof-of-concept or exploit code, if available
5. Potential impact, including how an attacker could exploit it

NVIDIA PSIRT acknowledges reports, assesses severity, coordinates a fix and
disclosure timeline with the reporter, and publishes security bulletins at
[https://www.nvidia.com/en-us/security/](https://www.nvidia.com/en-us/security/).

## Security Architecture & Context

**Project:** `tensorrtllm_backend` provides the Triton Inference Server
backend, ensemble/BLS model templates, documentation, and container
build recipe for serving [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
engines with in-flight batching.

**Software classification:** Library / Service component. This repository
contains no executable service logic of its own: the C++ backend, Python
pre/post-processing models, launch scripts, and example clients live in the
TensorRT-LLM project and are copied into the image by
`dockerfile/Dockerfile.triton.trt_llm_backend`. That Dockerfile does not use this
repository's `tensorrt_llm` submodule: it clones `TENSORRTLLM_REPO` (default
`NVIDIA/TensorRT-LLM`) at `TENSORRTLLM_REPO_TAG`, copies the scripts, models,
client, tools and examples from that clone, and installs the `tensorrt_llm`
wheel separately (`TENSORRTLLM_VER`). The backend runs inside
`tritonserver` and is exposed through Triton's HTTP/REST, gRPC, and metrics
endpoints.

**Primary security responsibility:** Faithfully load operator-supplied TensorRT
engines, tokenizers, and LoRA adapters; execute inference requests without
corrupting memory or leaking data between requests; and coordinate multi-GPU
and multi-node execution without exposing the coordination channel.

**Key security boundaries and interfaces:**

- Triton inference APIs (HTTP/REST, gRPC) carry untrusted request inputs such
  as `text_input`, `input_ids`, `lora_weights`, `lora_config`, and
  `guided_decoding_guide`.
- The model repository on disk (engines, `config.pbtxt`, tokenizer files, and
  Python model scripts) is read at load time.
- MPI (leader mode and orchestrator mode, `launch_triton_server.py`) links
  Triton processes across GPUs and nodes.
- The container build downloads TensorRT, PyTorch components, and
  TensorRT-LLM from NVIDIA-controlled sources.

**Repository Exposure Classification:** Public. Basis: the repository is
published on GitHub with public visibility.

**Service Exposure Classification:** Internal-Sensitive (confidence: medium).
Basis: the backend is normally deployed behind an operator's own network
controls, serves model inputs and outputs that may contain sensitive
user data, and does not itself provide authentication. This classification is
an assessment aid and not an official NVIDIA label.

## Threat Model

1. **Malicious or malformed inference inputs:** Crafted tensors sent to the
   `tensorrt_llm` model, such as oversized or inconsistent `input_ids`,
   `lora_weights`/`lora_config` shapes, or invalid `guided_decoding_guide`
   grammars (JSON schema, regex, EBNF), could trigger crashes, excessive
   resource use, or memory errors in the backend and in the TensorRT-LLM
   executor it wraps.
2. **Untrusted model artifacts:** Engines, tokenizer files, LoRA weights, and
   the Python model scripts in a model repository are loaded and executed with
   the privileges of the Triton process. A tampered or untrusted model
   repository, or a tokenizer or adapter fetched from a public hub, can lead
   to code execution or data disclosure.
3. **Unauthenticated MPI coordination channel:** In leader and orchestrator
   mode, Triton ranks communicate over MPI. A network-adjacent attacker, or a
   co-tenant on the same node or cluster, who can reach that channel could
   interfere with or observe inter-rank traffic.
4. **Supply-chain compromise at build time:** The Dockerfile downloads the
   TensorRT tarball over HTTPS, installs Python wheels from public and NVIDIA
   package indexes, runs an installer script fetched from a branch URL, and
   clones TensorRT-LLM at `release/1.2.1` by default, which is a release branch
   that can move, not a fixed tag. A compromised upstream, mutable reference, or
   package confusion could introduce malicious code into the image.
5. **Cross-request information disclosure:** Features that share state across
   requests, such as the LoRA cache keyed by `lora_task_id`, KV-cache reuse,
   and returned logits and performance metrics, could expose one tenant's
   data or adapters to another if deployed in a shared, multi-tenant setting.
6. **Resource exhaustion (denial of service):** Large batches, long
   sequences, streaming requests, and many LoRA adapters can exhaust GPU
   memory, host memory, or request queues, and a failure of a single MPI rank
   can stall the whole deployment.

## Critical Security Assumptions

- **Authentication and authorization are external.** Neither this repository
  nor the backend authenticates clients; operators must place Triton behind
  an authenticated, TLS-terminating gateway or a trusted network.
- **The model repository is trusted.** Engines, tokenizers, adapters, and
  Python model code are assumed to come from a trusted source and to be
  integrity-checked by the operator before being loaded.
- **The MPI fabric is private.** Inter-process and inter-node communication
  is assumed to run on an isolated, trusted network with no untrusted peers.
- **Inputs are validated upstream where limits matter.** Request size,
  sequence length, and rate limits are assumed to be enforced by Triton
  configuration or a front-end proxy.
- **The deployment is single-tenant, or tenants are isolated.** Shared caches
  and adapters are assumed not to cross trust boundaries.
- **Upstream components are trusted.** Triton Server, TensorRT,
  TensorRT-LLM, PyTorch, CUDA, and the base container images are assumed to
  be free of vulnerabilities in the pinned versions; consult the
  corresponding projects for their own security policies and advisories.
- **Build inputs are reviewed.** Operators building the image are expected to
  review and, where required, pin build arguments and download sources.
