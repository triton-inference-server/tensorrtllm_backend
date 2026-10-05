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

NVIDIA is dedicated to the security and trust of our software products and services, including all source code repositories managed through our organization.

To report a potential security vulnerability, please use one of the following channels:

1. **NVIDIA Vulnerability Disclosure Program** (preferred): https://www.nvidia.com/en-us/security/
2. **Web form:** [Security Vulnerability Submission Form](https://www.nvidia.com/object/submit-security-vulnerability.html)
3. **Email:** [NVIDIA PSIRT](mailto:psirt@nvidia.com). Please encrypt sensitive reports with NVIDIA's [PGP key](https://www.nvidia.com/en-us/security/pgp-key).
4. **GitHub Private Vulnerability Reporting (where enabled):** use the "Report a vulnerability" button on the Security tab of this repository.

**Do not open a public issue or pull request to report a vulnerability.**

Please include:

* Product or component name and version or branch
* Type of vulnerability
* Steps to reproduce
* Proof of concept, if available
* Potential impact and how it could be exploited

See https://www.nvidia.com/en-us/security/ for past NVIDIA Security Bulletins and Notices.

## Security Architecture and Context

**Project:** The Triton TensorRT-LLM Backend.

**Software type:** Software component (library, backend, client or tool) used as part of a Triton Inference Server deployment.

**Security boundaries:** The main security boundary is between this component and the data, models and configuration it is given, and between it and the server or application that hosts it.

**Repository Exposure Classification:** Public.

**Service Exposure Classification:** Deployment-dependent. Exposure depends on how the software is deployed and configured by the operator.

## Threat Model

1. **Untrusted input:** Requests, models, configuration or data supplied to this component may be malformed or malicious, and could cause crashes, memory errors or unintended behavior if not validated.
2. **Supply chain:** Source and build dependencies fetched at build or install time may be compromised, outdated or unpinned.
3. **Network exposure:** When deployed behind a network-facing server, endpoints may be reachable by untrusted clients. This component does not by itself provide authentication, authorization or encryption.
4. **Resource exhaustion:** Oversized or numerous requests may consume memory, compute or other resources and degrade availability.
5. **Information disclosure:** Logs, metrics and error messages may reveal sensitive data such as paths, identifiers or request content.

## Critical Security Assumptions

* The component is deployed in a trusted environment or behind a gateway that provides authentication, authorization, TLS and rate limiting.
* Models, configuration and other inputs come from trusted sources.
* Dependencies and the build environment are kept up to date and obtained from trusted sources.
* Operators protect secrets, certificates and credentials, and restrict access to logs and metrics.
* Host operating system, driver and hardware security are the operator's responsibility.
