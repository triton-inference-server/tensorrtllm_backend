import java.lang.InterruptedException
import groovy.transform.Field

// LLM repository configuration
BACKEND_REPO = "https://gitlab-master.nvidia.com/ftp/tekit_backend.git"
BACKEND_DEFAULT_BRANCH = "main"
BACKEND_ROOT = "backend"
BACKEND_DOCKER_IMAGE = "urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm:dev-triton-23.10-trt9.2.0.5-2"

// TURTLE repository configuration
TURTLE_REPO = "https://gitlab-master.nvidia.com/TensorRT/Infrastructure/turtle.git"
TURTLE_BRANCH = "25261fe0ad132419f7859bd9945cb0635ddfce37" // v5.0.4
TURTLE_ROOT = "turtle"

CASE_TO_EXAMPLE = [
  "gpt": "gpt",
  "opt": "opt",
  "llama": "llama",
  "mistral": "mistral",
  "mistral-ib": "mistral-ib",
  "gptj": "gptj",
  "gpt-ib": "gpt-ib",
  "gpt-ib-streaming": "gpt-ib",
  "gpt-ib-ptuning": "gpt-ib"
]

CASE_TO_MODEL = [
  "gpt": "gpt2",
  "opt": "opt-125m",
  "llama": "llama-models/llama-7b-hf",
  "mistral": "mistral-7b-v0.1",
  "mistral-ib": "mistral-7b-v0.1",
  "gptj": "gpt-j-6b",
  "gpt-ib": "gpt2",
  "gpt-ib-streaming": "gpt2",
  "gpt-ib-ptuning": "gpt2",
  "gpt-speculative-decoding": "gpt2"
]

CASE_TO_ENGINE_DIR = [
  "gpt": "gpt/trt_engine/gpt2/fp16/1-gpu/",
  "opt": "opt/trt_engine/opt-125m/fp16/1-gpu/",
  "llama": "llama/llama_outputs",
  "mistral": "llama/mistral_7b_outputs",
  "mistral-ib": "llama/ib_mistral_7b_outputs",
  "gptj": "gptj/gpt_outputs",
  "gpt-ib": "gpt/trt_engine/gpt2-ib/fp16/1-gpu/",
  "gpt-ib-streaming": "gpt/trt_engine/gpt2-ib/fp16/1-gpu/",
  "gpt-ib-ptuning": "gpt/trt_engine/email_composition/fp16/1-gpu/"
]

// Utilities
def checkoutSource(String repo, String branch, String directory) {
    def extensionsList = [
        [$class: 'CleanCheckout'],
        [$class: 'RelativeTargetDirectory',
            relativeTargetDir: directory],
        [$class: 'SubmoduleOption',
            parentCredentials: true,
            recursiveSubmodules: true,
            timeout: 300
            ]
    ]
    def scmSpec = [
        $class: "GitSCM",
        doGenerateSubmoduleConfigurations: false,
        submoduleCfg: [],
        branches: [[name: branch]],
        userRemoteConfigs: [
            [
                credentialsId: "svc_tensorrt_gitlab_api_token",
                name: "origin",
                url: repo,
            ]
        ],
        extensions: extensionsList,
    ]
    echo "Cloning with SCM spec: ${scmSpec.toString()}"
    checkout(scm: scmSpec, changelog: true)
}

def createKubernetesPodConfig(image, type)
{
    def targetCould = "kubernetes-cpu"
    def selectors = """
                  nvidia.com/node_type: builder
                  kubernetes.io/os: linux"""
    def containerConfig = ""

    switch(type)
    {
    case "agent":
        containerConfig = """
                  - name: alpine
                    image: alpine
                    command: ['cat']
                    tty: true
                    resources:
                      requests:
                        cpu: '2'
                        memory: 1Gi
                        ephemeral-storage: 5Gi
                      limits:
                        cpu: '2'
                        memory: 1Gi
                        ephemeral-storage: 5Gi
                    imagePullPolicy: Always"""
        break
    case "build":
        containerConfig = """
                  - name: trt-llm-backend
                    image: ${image}
                    command: ['cat']
                    volumeMounts:
                    - name: dshm
                      mountPath: /dev/shm
                    tty: true
                    resources:
                      requests:
                        cpu: '16'
                        memory: 64Gi
                        ephemeral-storage: 100Gi
                      limits:
                        cpu: '16'
                        memory: 64Gi
                        ephemeral-storage: 100Gi
                    imagePullPolicy: Always"""
        break
    default:
        def hasMultipleGPUs = (type.indexOf("x8") > -1)
        def gpuCount =  hasMultipleGPUs? "8" : "1"
        // Increase memorySize to 96GB to run C++ tests for GPT-J
        def memorySize = hasMultipleGPUs ? "320Gi" : "96Gi"
        def storageSize = hasMultipleGPUs ? "2000Gi" : "200Gi"

        targetCould = "kubernetes"
        selectors = """
                  nvidia.com/node_type: gpu_tester
                  kubernetes.io/os: linux
                  nvidia.com/gpu_type: ${type}
                  nvidia.com/driver_version: '535.104.05'"""

        containerConfig = """
                  - name: trt-llm-backend
                    image: ${image}
                    command: ['cat']
                    tty: true
                    resources:
                      requests:
                        cpu: '16'
                        memory: ${memorySize}
                        nvidia.com/gpu: ${gpuCount}
                        ephemeral-storage: ${storageSize}
                      limits:
                        cpu: '16'
                        memory: ${memorySize}
                        nvidia.com/gpu: ${gpuCount}
                        ephemeral-storage: ${storageSize}
                    imagePullPolicy: Always
                    volumeMounts:
                    - name: dshm
                      mountPath: /dev/shm
                    - name: scratch-trt-llm-data
                      mountPath: /home/scratch.trt_llm_data
                    securityContext:
                      capabilities:
                        add:
                        - SYS_ADMIN"""
        break
    }

    def podConfig = [
        cloud: targetCould,
        namespace: "sw-tensorrt",
        yaml: """
            apiVersion: v1
            kind: Pod
            spec:
                qosClass: Guaranteed
                nodeSelector: ${selectors}
                containers:
                  ${containerConfig}
                  - name: jnlp
                    image: jenkins/inbound-agent:3107.v665000b_51092-4
                    args: ['\$(JENKINS_SECRET)', '\$(JENKINS_NAME)']
                    resources:
                      requests:
                        cpu: '2'
                        memory: 5Gi
                        ephemeral-storage: 25Gi
                      limits:
                        cpu: '2'
                        memory: 5Gi
                        ephemeral-storage: 25Gi
                qosClass: Guaranteed
                volumes:
                - name: dshm
                  emptyDir:
                    medium: Memory
                - name: scratch-trt-llm-data
                  nfs:
                    server: 10.117.145.14
                    path: /vol/scratch1/scratch.michaeln_blossom
        """.stripIndent(),
    ]

    return podConfig
}

def uploadArtifacts(patternAbsPath, target)
{
    // Step 3: upload test results and logs to artifactory
    withCredentials([usernamePassword(credentialsId: 'urm-artifactory-creds', usernameVariable: 'SVC_TENSORRT_USER', passwordVariable: 'ARTIFACTORY_PASS'),])
    {
        rtServer (
            id: 'Artifactory',
            url: 'https://urm.nvidia.com/artifactory',
            // If you're using username and password:
            username: 'svc_tensorrt',
            password: "$ARTIFACTORY_PASS",
            // If Jenkins is configured to use an http proxy, you can bypass the proxy when using this Artifactory server:
            bypassProxy: true,
            // Configure the connection timeout (in seconds).
            // The default value (if not configured) is 300 seconds:
            timeout: 300
        )
        rtUpload (
            serverId: 'Artifactory',
            spec: """{
                "files": [
                    {
                    "pattern": "${patternAbsPath}",
                    "target": "${target}"
                    }
                ]
            }""",
        )
    }
}


def runBuild()
{
    sh "pwd && ls -alh"
    sh "env"
    sh "timeout 30 mount || true"
    // allow to checkout from forked repo, svc_tensorrt needs to have access to the repo, otherwise clone will fail
    // Step 1: cloning tekit source code
    checkoutSource(env.gitlabSourceRepoHttpUrl ? env.gitlabSourceRepoHttpUrl: BACKEND_REPO, env.gitlabBranch ? env.gitlabBranch : BACKEND_DEFAULT_BRANCH, BACKEND_ROOT)
    sh "cd ${BACKEND_ROOT} && git config --unset core.hooksPath"
    sh "cd ${BACKEND_ROOT} && git lfs install"
    sh "cd ${BACKEND_ROOT} && git submodule update --init --recursive"
    container("trt-llm-backend") {
      // Step 2: checking code style
      sh "pip3 install pre-commit"
      sh "git config --global --add safe.directory \$(realpath ${BACKEND_ROOT})"
      sh "cd ${BACKEND_ROOT} && pre-commit run -a"
      // Step 3: packaging tensorrt-llm backend
      sh "rm -rf tensorrt_llm_backend"
      sh "cp -r ${BACKEND_ROOT} tensorrt_llm_backend"
      sh "cp ${BACKEND_ROOT}/scripts/package_trt_llm_backend.sh package_trt_llm_backend.sh"
      sh "bash package_trt_llm_backend.sh tensorrt_llm_backend.tar.gz tensorrt_llm_backend"
    }

    uploadArtifacts("tensorrt_llm_backend.tar.gz", "sw-tensorrt-generic/llm-artifacts/${JOB_NAME}/${BUILD_NUMBER}/")

    container("trt-llm-backend") {
      // Step 4: build tensorrt-llm backend
      sh "cd ${BACKEND_ROOT} && python3 tensorrt_llm/scripts/build_wheel.py --trt_root /usr/local/tensorrt"
      sh "cd ${BACKEND_ROOT} && mkdir tensorrt_llm/cpp/tensorrt_llm/batch_manager/x86_64-linux-gnu/"
      sh "cd ${BACKEND_ROOT} && mv tensorrt_llm/cpp/build/tensorrt_llm/batch_manager/libtensorrt_llm_batch_manager_static.a tensorrt_llm/cpp/tensorrt_llm/batch_manager/x86_64-linux-gnu/libtensorrt_llm_batch_manager_static.pre_cxx11.a"
      sh "cd ${BACKEND_ROOT}/inflight_batcher_llm && patch -N CMakeLists.txt -i CMakeLists.txt.patch"
      sh "cd ${BACKEND_ROOT}/inflight_batcher_llm && bash scripts/build.sh"
      sh "tar -zcf tensorrt_llm_backend_internal.tar.gz ${BACKEND_ROOT}"
    }
    // Step 5: upload package to artifactory
    uploadArtifacts("tensorrt_llm_backend_internal.tar.gz", "sw-tensorrt-generic/llm-artifacts/${JOB_NAME}/${BUILD_NUMBER}/")
}

def installDependency()
{
  sh "pwd && ls -alh"
  sh "env"
  sh "timeout 30 mount || true"
  // Step 1: create BACKEND_ROOT dir and clone TURTLE
  checkoutSource(TURTLE_REPO, TURTLE_BRANCH, TURTLE_ROOT)

  container("trt-llm-backend") {
    def backendTarfile = "https://urm.nvidia.com/artifactory/sw-tensorrt-generic/llm-artifacts/${JOB_NAME}/${BUILD_NUMBER}/tensorrt_llm_backend_internal.tar.gz"

    sh "curl -L ${backendTarfile} | tar -xz ${BACKEND_ROOT}"
    sh "ls -lah"
    sh "cd ${BACKEND_ROOT} && pip3 install --extra-index-url https://pypi.ngc.nvidia.com tensorrt_llm/build/tensorrt_llm*.whl"

    sh "env"
    sh "which python3"
    sh "python3 --version"
    sh "apt-get update"
    sh "apt-get install -y libffi-dev"
  }
}

def runTRTLLMBackendTest(caseName)
{
  container("trt-llm-backend") {

    def modelPath = "/home/scratch.trt_llm_data/llm-models/" + CASE_TO_MODEL[caseName]
    def tokenizerType = "auto"
    def backendPath = sh (script: "realpath ${BACKEND_ROOT}",returnStdout: true).trim()

    sh "ps -aux"
    sh "nvidia-smi"
    sh "rm -rf /opt/tritonserver/backends/tensorrtllm"

    if (caseName.contains("-ib") || caseName.contains("speculative-decoding")) {
      sh "mkdir /opt/tritonserver/backends/tensorrtllm"
      sh "cd ${BACKEND_ROOT} && cp inflight_batcher_llm/build/libtriton_tensorrtllm.so /opt/tritonserver/backends/tensorrtllm"
    }

    if (caseName.contains("llama")) {
      tokenizerType = "llama"
    }

    if (caseName.contains("speculative-decoding")) {
      catchError(buildResult: 'FAILURE', stageResult: 'FAILURE')
      {
        sh "cd ${BACKEND_ROOT} && bash tests/build_model.sh gpt-ib"
        sh "cd ${BACKEND_ROOT} && bash tests/build_model.sh gpt-medium-ib"
        sh "cd ${BACKEND_ROOT} && tests/test.sh gpt-speculative-decoding ${backendPath}/tensorrt_llm/examples/gpt/trt_engine/gpt2-medium-ib/fp16/1-gpu/ ${modelPath} ${tokenizerType} ${backendPath}/tensorrt_llm/examples/gpt/trt_engine/gpt2-ib/fp16/1-gpu/"
      }
    } else {
      def buildExample = CASE_TO_EXAMPLE[caseName]
      def testExample = CASE_TO_EXAMPLE[caseName]
      def enginePath = "${backendPath}/tensorrt_llm/examples/" + CASE_TO_ENGINE_DIR[caseName]

      if (caseName.contains("-ptuning")) {
        buildExample += "-ptuning"
        testExample += "-ptuning"
      }

      if (caseName.contains("-streaming")) {
        testExample += "-streaming"
      }

      catchError(buildResult: 'FAILURE', stageResult: 'FAILURE')
      {
        sh "cd ${BACKEND_ROOT} && bash tests/build_model.sh ${buildExample}"
        sh "cd ${BACKEND_ROOT} && bash tests/test.sh ${testExample} ${enginePath} ${modelPath} ${tokenizerType}"
      }
    }
  }
}

def runLLMBackendTestTURTLE(platform, testList, perfMode=false, timeout=0)
{

    def backendPath = sh (script: "realpath ${BACKEND_ROOT}",returnStdout: true).trim()
    def turtleBinPath = sh (script: "realpath ${TURTLE_ROOT}/bin/turtle",returnStdout: true).trim()

    // Step 1: run TURTLE tests
    container("trt-llm-backend") {
        script
        {
          // Step 1.1: setup python environments
          sh "timeout 30 mount || true"
          sh "ps -aux"
          sh "nvidia-smi"

          sh "rm -rf ${platform}-${testList}.tar.gz ${platform}-${testList}/*"
          sh "ls -lah"

          def turtleCmdLine = [
              "LLM_BACKEND_ROOT=${backendPath}",
              "LLM_MODELS_ROOT=/home/scratch.trt_llm_data/llm-models",
              "python3",
              turtleBinPath,
              "--test-prefix ${platform}",
              "-D ${backendPath}/tests/llm-backend-test-defs/turtle/defs",
              "-f ${backendPath}/tests/llm-backend-test-defs/turtle/test_lists/bloom/${testList}.txt",
              "--waives-file ${backendPath}/tests/llm-backend-test-defs/turtle/test_lists/waives.txt",
              "--test-timeout",
              "7200",
              "--test-python3-exe /usr/bin/python3",
              "--junit-xml",
              "--perf",
              "--perf-clock-gpu-configs-file ${backendPath}/tests/llm-backend-test-defs/turtle/perf_configs/gpu_configs.yml",
              "--perf-log-formats csv",
              "--perf-log-formats yaml",
              "--output-dir ${platform}-${testList}/"
          ]

          if (timeout > 0)
          {
              turtleCmdLine += ["--session-timeout", "${timeout}"]
          }

          // Step 1.2: launch TURTLE session
          catchError(buildResult: 'FAILURE', stageResult: 'FAILURE')
          {
              sh turtleCmdLine.join(" ")
          }

          sh "tar -czvf results-${platform}-${testList}.tar.gz ${platform}-${testList}/"
        }
    }

    // Step 2: upload test results and logs to artifactory
    uploadArtifacts("results-${platform}-${testList}.tar.gz", "sw-tensorrt-generic/llm-artifacts/${JOB_NAME}/${BUILD_NUMBER}/test-results/")
}

pipeline {
    agent {
      kubernetes createKubernetesPodConfig("", "agent")
    }
    options {
      // Check the valid options at: https://www.jenkins.io/doc/book/pipeline/syntax/
      // some step like results analysis stage, does not need to check out source code
      skipDefaultCheckout()
      // to better analyze the time for each step/test
      timestamps()
    }
    environment {
      //Workspace normally is: /home/jenkins/agent/workspace/LLM/L0_MergeRequest@tmp/
      HF_HOME="${env.WORKSPACE_TMP}/.cache/huggingface"
    }
    post {
        unsuccessful {
          updateGitlabCommitStatus name: "Jenkins build", state: "failed"
        }
        success {
          updateGitlabCommitStatus name: "Jenkins build", state: "success"
        }
        aborted {
          updateGitlabCommitStatus name: "Jenkins build", state: 'canceled'
        }
    }
    stages {
      stage("Update status")
      {
        steps {
          updateGitlabCommitStatus name: "Jenkins build", state: 'running'
        }
      }
      stage("Build") {
        agent {
          kubernetes createKubernetesPodConfig(BACKEND_DOCKER_IMAGE, "build")
        }
        steps {
          runBuild()
        }
      }
      stage("Test")
      {
        parallel {
          stage("A30 Python tester") {
            agent {
              kubernetes createKubernetesPodConfig(BACKEND_DOCKER_IMAGE, "A30")
            }
            stages {
              stage("Setup tester") {
                steps {
                  installDependency()
                }
              }
              stage("Test gpt") {
                steps {
                  runTRTLLMBackendTest("gpt")
                }
              }
              stage("Test opt") {
                steps {
                  runTRTLLMBackendTest("opt")
                }
              }
              stage("Test llama") {
                steps {
                  runTRTLLMBackendTest("llama")
                }
              }
              stage("Test gptj") {
                steps {
                  runTRTLLMBackendTest("gptj")
                }
              }
              stage("Test mistral") {
                steps {
                  runTRTLLMBackendTest("mistral")
                }
              }
              stage("Test l0_functional") {
                steps {
                  runLLMBackendTestTURTLE("A30", "l0_functional")
                }
              }
            }
          }
          stage("A30 CPP tester") {
            agent {
              kubernetes createKubernetesPodConfig(BACKEND_DOCKER_IMAGE, "A30")
            }
            stages {
              stage("Setup tester") {
                steps {
                  installDependency()
                }
              }
              stage("Test mistral-ib") {
                steps {
                  runTRTLLMBackendTest("mistral-ib")
                }
              }
              stage("Test gpt-ib-streaming") {
                steps {
                  runTRTLLMBackendTest("gpt-ib-streaming")
                }
              }
            }
          }
          stage("A100_80GB_PCIE CPP tester") {
            agent {
              kubernetes createKubernetesPodConfig(BACKEND_DOCKER_IMAGE, "A100_80GB_PCIE")
            }
            stages {
              stage("Setup tester") {
                steps {
                  installDependency()
                }
              }
              stage("Test gpt-ib") {
                steps {
                  runTRTLLMBackendTest("gpt-ib")
                }
              }
              stage("Test gpt-ib-ptuning") {
                steps {
                  runTRTLLMBackendTest("gpt-ib-ptuning")
                }
              }
              stage("Test gpt-speculative-decoding") {
                steps {
                  runTRTLLMBackendTest("gpt-speculative-decoding")
                }
              }
            }
          }
        }
      }
      stage("Check Test Results") {
          agent {
            kubernetes createKubernetesPodConfig("", "agent")
          }
          steps {
            container("alpine") {
              sh "rm -rf A30* *.tar.gz"
              sh "wget -nv https://urm.nvidia.com/artifactory/sw-tensorrt-generic/llm-artifacts/${JOB_NAME}/${BUILD_NUMBER}/test-results/results-A30-l0_functional.tar.gz || true"
              sh "find . -name results-\\*.tar.gz -type f -exec tar -zxvf {} \\; || true"
            }
            junit '**/results.xml'
          }
      } // Collect test result stage
    } // stages
} // pipeline
