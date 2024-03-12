import groovy.json.JsonOutput

// LLM-Backend repository configuration
BACKEND_REPO = "https://gitlab-master.nvidia.com/ftp/tekit_backend.git"
BACKEND_BRANCH = "main"
BACKEND_ROOT = "backend"
BACKEND_SBSA_DOCKER_IMAGE = "urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm:tritonserver-24.01-py3-aarch64-ubuntu22.04-trt9.3.0.1-src_non_cxx11_abi-devel-202403061430-472"

// TURTLE repository configuration
TURTLE_REPO = "https://gitlab-master.nvidia.com/TensorRT/Infrastructure/turtle.git"
TURTLE_BRANCH = "main"
TURTLE_ROOT = "turtle"

// TODO: Move common variables to an unified location
BUILD_CORES = "16"
CCACHE_DIR = "/mnt/sw-tensorrt-pvc/scratch.trt_ccache/llm_ccache"

// Utilities
def checkoutSource(String repo, String branch, String directory) {
    def extensionsList = [
        [$class: 'CleanCheckout'],
        [$class: 'RelativeTargetDirectory',
            relativeTargetDir: directory],
        [$class: 'SubmoduleOption',
            parentCredentials: true,
            recursiveSubmodules: true,
            timeout: 60
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
    // Step 1: cloning tekit source code
    sh "pwd && ls -alh"
    sh "env"

    checkoutSource(BACKEND_REPO, env.gitlabBranch ? env.gitlabBranch : BACKEND_BRANCH, BACKEND_ROOT)

    if (env.dockerImage) {
        BACKEND_SBSA_DOCKER_IMAGE = env.dockerImage
    }

    def backendCommit = sh (script: "cd ${BACKEND_ROOT} && git rev-parse HEAD",returnStdout: true).trim()
    echo "Rewriting BACKEND_BRANCH from ${BACKEND_BRANCH} to ${backendCommit}..."
    BACKEND_BRANCH = backendCommit

    // WAR: PVC mount is not setup on GH200 machines, use a small local cache as a WAR
    docker.image(BACKEND_SBSA_DOCKER_IMAGE).inside(' -v /tmp/ccache:${CCACHE_DIR}:rw') {
        // Random sleep to avoid resource contention
        sleep(10 * Math.random())
        sh "nproc && free -g && hostname"
        sh "ccache -M 10Gi"
        sh "cat ${CCACHE_DIR}/ccache.conf"

        sh "env"
        sh "ldconfig --print-cache || true"
        sh "ls -lh /"
        sh "id || true"
        sh "whoami || true"

        // Step 2: checking code style
        sh "cd ${BACKEND_ROOT} && git config --unset core.hooksPath"
        sh "cd ${BACKEND_ROOT} && git lfs install"
        sh "cd ${BACKEND_ROOT} && git submodule update --init --recursive"

        sh "pip3 install pre-commit"
        sh "git config --global --add safe.directory \$(realpath ${BACKEND_ROOT})"
        sh "cd ${BACKEND_ROOT} && pre-commit run -a"
        // Step 3: packaging tensorrt-llm backend
        sh "rm -rf tensorrt_llm_backend"
        sh "cp -r ${BACKEND_ROOT} tensorrt_llm_backend"
        sh "cp ${BACKEND_ROOT}/scripts/package_trt_llm_backend.sh package_trt_llm_backend.sh"
        sh "bash package_trt_llm_backend.sh tensorrt_llm_backend_aarch64.tar.gz tensorrt_llm_backend"
    }
    uploadArtifacts("tensorrt_llm_backend_aarch64.tar.gz", "sw-tensorrt-generic/llm-artifacts/${hostJobName}/${hostBuildNumber}/")

    docker.image(BACKEND_SBSA_DOCKER_IMAGE).inside(' -v /tmp/ccache:${CCACHE_DIR}:rw') {
        // Step 4: build tensorrt-llm backend
        sh "cd ${BACKEND_ROOT} && python3 tensorrt_llm/scripts/build_wheel.py --use_ccache -j ${BUILD_CORES} -a '90-real' --trt_root /usr/local/tensorrt"
        sh "cd ${BACKEND_ROOT}/inflight_batcher_llm && bash scripts/build.sh -u"
        sh "tar -zcf tensorrt_llm_backend_internal_aarch64.tar.gz ${BACKEND_ROOT}"
    }
    uploadArtifacts("tensorrt_llm_backend_internal_aarch64.tar.gz", "sw-tensorrt-generic/llm-artifacts/${hostJobName}/${hostBuildNumber}/")

}


pipeline {
    agent {
        label 'sbsa-a100-80gb-pcie-x4||sbsa-gh200-480gb'
    }
    options {
        // Check the valid options at: https://www.jenkins.io/doc/book/pipeline/syntax/
        // some step like results analysis stage, does not need to check out source code
        skipDefaultCheckout()
        // to better analyze the time for each step/test
        timestamps()
        timeout(time: 8, unit: 'HOURS')
    }
    environment
    {
        //Workspace normally is: /home/jenkins/agent/workspace/LLM-Backend/L0_MergeRequest@tmp/
        HF_HOME="${env.WORKSPACE_TMP}/.cache/huggingface"
        CCACHE_DIR="${CCACHE_DIR}"
    }
    stages {
        stage('Prepare') {
            steps {
                sh 'pwd'
                sh 'ls -lah'
                sh 'rm -rf ./*'
                sh 'ls -lah'
                echo "hostJobName: ${env.hostJobName}"
                echo "hostBuildNumber: ${env.hostBuildNumber}"
                echo "dockerImage: ${env.dockerImage}"
                echo "gitlabBranch: ${env.gitlabBranch}"
            }
        }
         // Build stage
        stage("Build") {
            steps {
                runBuild()
            }
        }
    } // stages
} // pipeline
