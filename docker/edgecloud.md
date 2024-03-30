# Edge Cloud

## Current Containers
Option1: Ubuntu20.04 based on nvcr.io/nvidia/tritonserver:22.03-py3, you can build the container based on the docker/Dockerfile.triton 

Option2: Ubuntu22.04 based on nvidia/cuda:11.7.1-devel-ubuntu22.04, you can build the container in the following two steps:
 * Build one base container based on Dockerfile: docker/Dockerfile.ubuntu22cu117, the created image named "myros2ubuntu22cuda117:latest"
 * Run build-image.sh (call docker/Dockerfile.ros2humble) to create a ROS2 container (tagged as "myros2humble:latest") based on the "myros2ubuntu22cuda117:latest"

 Option2 version needs additional steps to prevent build error in isaac_ros_nitros (error: ‘unique_lock’ is not a member of ‘std’). Follow the changes [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros/pull/8/commits/0e243982f6a6c69ef896b4c621f422d170760825) 

After the docker container image is built, run the follow script to start the container
```bash
./scripts/runcontainer.sh myros2humble:latest
```

## Docker
Install [Docker](https://docs.docker.com/engine/install/ubuntu/) and follow [Post-installation steps for Linux](https://docs.docker.com/engine/install/linux-postinstall/)

Setup Docker and nvidia container runtime via [nvidiacontainer1](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) [nvidiacontainer2](https://docs.nvidia.com/dgx/nvidia-container-runtime-upgrade/index.html
)


After you build the container, you can check the new container via "docker images", note down the image id, and run this image:
```bash
sudo docker run -it --rm 486a56765aad
```
After you entered the container and did changes inside the container, click "control+P+Q" to exit the container without terminate the container. Use "docker ps" to check the container id, then use "docker commit" to commit changes:
```bash
docker commit -a "Kaikai Liu" -m "First ROS2-x86 container" 196073a381b4 myros2:v1
```
Now, you can see your newly created container image named "myros2:v1" in "docker images".

You can now start your ROS2 container (i.e., myros2:v1) via runcontainer.sh, change the script file if you want to change the path of mounted folders. 
```bash
sudo xhost +si:localuser:root
./scripts/runcontainer.sh [containername]
```
after you 
Re-enter a container: use the command "docker exec -it container_id /bin/bash" to get a bash shell in the container.

Popular Docker commands:
 * Stop a running container: docker stop container_id
 * Stop all containers not running: docker container prune
 * Delete docker images: docker image rm dockerimageid

## Container Installation
Check the Docker section for detailed information.

Use the Dockerfile under scripts folder to build the container image:
```bash
myROS2/docker$ docker build -t myros2ubuntu22cuda117 .
```
You can also build the docker image via docker vscode extension. After the extension is installed, simply right click the Dockerfile and select "build image"

Enter ROS2 container (make sure the current directory is myROS, it will be mounted to the container)
```bash
MyRepo/myROS2$ ./scripts/runcontainer.sh myros2ubuntu22cuda117
```

## K3S for GPUs

the required steps are first, installing the NVIDIA drivers on the K3s node. 

The second step is to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html), which helps to expose the GPU resources from the K3S node to the containers running on it. The third step is to tell K3S to use this toolkit to enable GPUs on the containers. One point to pay attention to is to install only the containerd version of the toolkit. K3S does not use Docker at all, since Kubernetes already deprecated Docker, and it uses only the containerd to manage containers. Installing also the Docker support won’t impact how your cluster works since it will also implicitly install the containerd support, but since we avoid installing unnecessary packages on our lean Kubernetes nodes, we directly go with the containerd installation. ()


Step 3: Configure K3S to use nvidia-container-runtime. Tell K3S to use nvidia-container-runtime (which is a kind of plugin of containerd) on the containerd of our node. K3D's tutorial (https://k3d.io/usage/guides/cuda/#configure-containerd). The only part we are interested in in that guide is the “Configure containerd” section. The template they have shared is configuring the containerd to use the nvidia-container-runtime plugin, together with a couple of more extra boilerplate settings. To install that template to our node, we can simply run the following command:

```bash
    sudo wget https://k3d.io/v4.4.8/usage/guides/cuda/config.toml.tmpl -O /var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl
```

Step 4: Install NVIDIA device plugin for Kubernetes. The [NVIDIA device plugin for Kubernetes](https://github.com/NVIDIA/k8s-device-plugin) is a DaemonSet that scans the GPUs on each node and exposes them as GPU resources to our Kubernetes nodes.

If you follow the documentation of the device plugin, there is also a Helm chart available to install it. On K3S, we have a simple Helm controller that allows us to install Helm charts on our cluster. Let us leverage it and deploy this Helm chart:

```bash
    cat <<EOF | kubectl apply -f -
    apiVersion: helm.cattle.io/v1
    kind: HelmChart
    metadata:
    name: nvidia-device-plugin
    namespace: kube-system
    spec:
    chart: nvidia-device-plugin
    repo: https://nvidia.github.io/k8s-device-plugin
    EOF
```
you can install the device plugin also by applying the manifest directly or by installing the chart using “helm install”.

Step 5: Test everything on a CUDA-enabled Pod. Finally, we can test everything by creating a Pod that uses the CUDA Docker image and requests a GPU resource:
```bash
    cat <<EOF | kubectl create -f -
    apiVersion: v1
    kind: Pod
    metadata:
    name: gpu
    spec:
    restartPolicy: Never
    containers:
        - name: gpu
        image: "nvidia/cuda:11.4.1-base-ubuntu20.04"
        command: [ "/bin/bash", "-c", "--" ]
        args: [ "while true; do sleep 30; done;" ]
        resources:
            limits:
            nvidia.com/gpu: 1
    EOF
```
Finally let us run the nvidia-smi on our Pod:
```bash
    kubectl exec -it gpu -- nvidia-smi
```
[Kubernets documentation](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)

## Terraform
[Terraform](https://developer.hashicorp.com/terraform) is an infrastructure as code (IaC) tool that allows you to define and manage cloud and on-premises resources in a human-readable configuration format. Terraform enables you to define your infrastructure using configuration files (written in HashiCorp Configuration Language or HCL). The core Terraform workflow involves three stages:
    * Write: Define resources across multiple cloud providers and services in your configuration.
    * Plan: Terraform generates an execution plan based on your configuration and existing infrastructure.
    * Apply: Execute the proposed operations (create, update, or destroy) in the correct order.

> Terraform works with various platforms and services through providers. Providers allow Terraform to interact with APIs of different cloud providers and services. Available Providers: There are thousands of publicly available providers on the Terraform Registry, including AWS, Azure, Google Cloud Platform (GCP), Kubernetes, GitHub, and more.

Download Terraform from [link](https://developer.hashicorp.com/terraform/install)
Install for Linux:
```
wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install terraform
```

[Terraform for Docker](https://developer.hashicorp.com/terraform/tutorials/docker-get-started)

## Waldur
[Waldur](https://waldur.com/) is a platform for managing private clouds, public clouds and HPC Centres. It is a hybrid cloud orchestrator used for cloud management, orchestration, and accounting. It automates OpenStack and Slurm, has billing, user self-service and support desk. Waldur MasterMind has the capability to manage Kubernetes (K8s) clusters, including lightweight distributions like K3s. Waldur supports the creation and management of Kubernetes clusters via Rancher and OpenStack. Features include:
    * User self-service portal.
    * Service Desk integration (Atlassian Service Desk).
    * Support for multiple authentication and authorization mechanisms.
    * Billing and fine-grained accounting.
    * Automated provisioning for OpenStack, SLURM, and Kubernetes clusters (using Rancher).

Use Terraform to provision the infrastructure (set up networking, and manage other infrastructure components) and then manage it using Waldur (manage the entire lifecycle of your K3s cluster, including user access, resource allocation, cost tracking, self-service portals, billing, and fine-grained accounting).