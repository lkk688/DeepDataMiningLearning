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
[K3s](https://docs.k3s.io/installation). Install K3s using the Installation Script via [K3squickstart](https://docs.k3s.io/quick-start). The installation script is the easiest way to set up K3s as a service on systemd and openrc based systems. Run the following command on the master node to install K3s and start the service automatically:
```bash
curl -sfL https://get.k3s.io | sh -
#After successful installation, verify the K3s service status using:
sudo systemctl restart k3s
sudo systemctl status k3s
```
Additional utilities will be installed, including kubectl, crictl, ctr, k3s-killall.sh, and k3s-uninstall.sh. A kubeconfig file will be written to /etc/rancher/k3s/k3s.yaml and the kubectl installed by K3s will automatically use it.

If you want to uninstall k3s, run the following script:
```bash
/usr/local/bin/k3s-agent-uninstall.sh #Uninstall K3s from Agent Nodes
/usr/local/bin/k3s-uninstall.sh
rm -rf /var/lib/rancher/k3s
```

You can also install and uninstall K3S via our script
```bash
lkk@dellr530:~/MyRepo/DeepDataMiningLearning/docker$ ./k3s-install.sh
```

A single-node server installation is a fully-functional Kubernetes cluster, including all the datastore, control-plane, kubelet, and container runtime components necessary to host workload pods. It is not necessary to add additional server or agents nodes, but you may want to do so to add additional capacity or redundancy to your cluster.
```bash
#check server address
kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}'
$ kubectl get nodes
NAME       STATUS   ROLES                  AGE   VERSION
dellr530   Ready    control-plane,master   24s   v1.28.8+k3s1
```

Remove MicroK8S
```bash
sudo microk8s reset
sudo snap remove microk8s
sudo microk8s disable
sudo microk8s status
```

You can verify by listing all the Kubernetes objects in the kube-system namespace.
```bash
$ kubectl get all -n kube-system
$ kubectl get pods --all-namespaces #check which containers (pods) get created
NAMESPACE     NAME                                      READY   STATUS      RESTARTS   AGE
kube-system   coredns-6799fbcd5-tgbv9                   1/1     Running     0          2m14s
kube-system   local-path-provisioner-6c86858495-gth54   1/1     Running     0          2m14s
kube-system   helm-install-traefik-crd-b7wgn            0/1     Completed   0          2m14s
kube-system   helm-install-traefik-vgs59                0/1     Completed   1          2m14s
kube-system   svclb-traefik-b18a0d17-f67d9              2/2     Running     0          108s
kube-system   metrics-server-54fd9b65b-cwvsm            1/1     Running     0          2m14s
kube-system   traefik-f4564c4f4-tsw86                   1/1     Running     0          108s
$ kubectl get pods
No resources found in default namespace.
```
We can see a basic K3s setup composed by:
    * Traefik as an ingress controller for HTTP reverse proxy and load balancing
    * CoreDns to manage DNS resolution inside the cluster and nodes
    * Local Path Provisioner provides a way to utilize the local storage in each node
    * Helm, which we can use to customize packaged components

Instead of running components in different processes, K3s will run all in a single server or agent process. As it is packaged in a single file, we can also work offline, using an Air-gap installation. Interestingly, we can also run K3s in Docker using K3d.

Test Nginx image with 2 replicas available on port 80:
```bash
$ kubectl create deployment nginx --image=nginx --port=80 --replicas=2
deployment.apps/nginx created
lkk@dellr530:~/MyRepo/DeepDataMiningLearning/docker$ kubectl get pods
NAME                     READY   STATUS    RESTARTS   AGE
nginx-7c5ddbdf54-5nczm   1/1     Running   0          22s
nginx-7c5ddbdf54-cc5wv   1/1     Running   0          22s
```
Pods are not permanent resources and get created and destroyed constantly. Therefore, we need a Service to map the pods’ IPs to the outer world dynamically. Services can be of different types. We'll choose a ClusterIp. In Kubernetes, a ClusterIP is a virtual IP address assigned to a Service. In Kubernetes, a Service is an abstract way to expose an application running on a set of Pods. Services allow clients (both inside and outside the cluster) to connect to the application. They provide load balancing across the different backing Pods.

A ClusterIP is a type of Service that has a cluster-scoped virtual IP address. Clients within the Kubernetes cluster can connect to this virtual IP address. Kubernetes then load-balances traffic to the Service across the different Pods associated with it.

```bash
kubectl create service clusterip nginx --tcp=80:80
kubectl describe service nginx
```
creates a ClusterIP Service named "nginx" that exposes port 80 within the Kubernetes cluster. kubectl create service is the command to create a Kubernetes service. clusterip specifies the type of service. In this case, it’s a ClusterIP Service. nginx is the name of the service being created. The port format is "--tcp=<external-port>:<internal-port>", 80:80 means that external traffic hitting port 80 will be directed to the Pods associated with this service on port 80.

We can see the Endpoints corresponding to the pods (or containers) addresses where we can reach our applications. Services don’t have direct access. An Ingress Controller is usually in front of them for caching, load balancing, and security reasons, such as filtering out malicious requests. Finally, let’s define a Traefik controller in a YAML file. This will route the traffic from the incoming request to the service:

```bash
lkk@dellr530:~$ nano traefik_nginx.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: nginx
  annotations:
    ingress.kubernetes.io/ssl-redirect: "false"
spec:
  rules:
  - http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: nginx
            port:
              number: 80
lkk@dellr530:~$ kubectl apply -f traefik_nginx.yaml
ingress.networking.k8s.io/nginx created
lkk@dellr530:~$ kubectl describe ingress nginx #describe our ingress controller:
Name:             nginx
Labels:           <none>
Namespace:        default
Address:          130.65.157.217
Ingress Class:    traefik
Default backend:  <default>
Rules:
  Host        Path  Backends
  ----        ----  --------
  *
              /   nginx:80 (10.42.0.10:80,10.42.0.9:80)
Annotations:  ingress.kubernetes.io/ssl-redirect: false
Events:       <none>
lkk@dellr530:~$ kubectl get pods,services,endpointslices
NAME                         READY   STATUS    RESTARTS   AGE
pod/nginx-7c5ddbdf54-5nczm   1/1     Running   0          88m
pod/nginx-7c5ddbdf54-cc5wv   1/1     Running   0          88m

NAME                 TYPE        CLUSTER-IP    EXTERNAL-IP   PORT(S)   AGE
service/kubernetes   ClusterIP   10.43.0.1     <none>        443/TCP   101m
service/nginx        ClusterIP   10.43.40.47   <none>        80/TCP    7m8s

NAME                                         ADDRESSTYPE   PORTS   ENDPOINTS              AGE
endpointslice.discovery.k8s.io/kubernetes    IPv4          6443    130.65.157.217         101m
endpointslice.discovery.k8s.io/nginx-k5cpb   IPv4          80      10.42.0.9,10.42.0.10   7m8s
```
Open the link of "130.65.157.217" can open the nginx server frontpage.

Access the NGINX Pod: find the name of the NGINX Pod you want to access. You can list all Pods in the current namespace using:
```bash
lkk@dellr530:~$ kubectl get pods
NAME                     READY   STATUS    RESTARTS   AGE
nginx-7c5ddbdf54-5nczm   1/1     Running   0          92m
nginx-7c5ddbdf54-cc5wv   1/1     Running   0          92m
lkk@dellr530:~$ kubectl exec -it nginx-7c5ddbdf54-5nczm -- /bin/bash
root@nginx-7c5ddbdf54-5nczm:/# cat /etc/nginx/nginx.conf
#If you made changes to the configuration, restart NGINX inside the Pod: service nginx restart
#exit the Pod by typing exit
```

Delete resources. If your Pods are managed by a Deployment (which is common in production environments), consider scaling down the Deployment to zero replicas:
```bash
kubectl scale deployment nginx --replicas=0
lkk@dellr530:~$ kubectl delete pods --all
pod "nginx-7c5ddbdf54-5nczm" deleted
pod "nginx-7c5ddbdf54-cc5wv" deleted
lkk@dellr530:~$ kubectl delete services --all
service "kubernetes" deleted
service "nginx" deleted
lkk@dellr530:~$ kubectl delete -f traefik_nginx.yaml
ingress.networking.k8s.io "nginx" deleted
lkk@dellr530:~$ kubectl delete endpointslice --all
endpointslice.discovery.k8s.io "kubernetes" deleted
lkk@dellr530:~$ kubectl get pods,services,endpointslices
```

Set up a Jupyter Lab pod in K3s cluster 
```bash
lkk@dellr530:~$ kubectl create deployment jupyter-lab --image=jupyter/base-notebook
deployment.apps/jupyter-lab created
lkk@dellr530:~$ kubectl get pods
NAME                          READY   STATUS              RESTARTS   AGE
jupyter-lab-d7bdfb78b-q4pj4   0/1     ContainerCreating   0          11s
lkk@dellr530:~$ kubectl expose deployment jupyter-lab --type=ClusterIP --port=8888 #Expose the Deployment as a Service:
service/jupyter-lab exposed
#To access Jupyter Lab from your local browser, you’ll need to forward the port
lkk@dellr530:~$ kubectl port-forward service/jupyter-lab 8888:8888
Forwarding from 127.0.0.1:8888 -> 8888
Forwarding from [::1]:8888 -> 8888
lkk@dellr530:~$ kubectl describe service jupyter-lab
lkk@dellr530:~$ kubectl exec -it pod/jupyter-lab-d7bdfb78b-q4pj4 -- /bin/bash
```
In 'image' part, The hostname of the container image registry (e.g., Docker Hub, Google Container Registry, etc.). If omitted, Kubernetes assumes the Docker public registry. Open your web browser and visit "http://130.65.157.217:8888". You should see the Jupyter Lab interface.


Install Helm
```bash
curl https://raw.githubusercontent.com/helm/helm/HEAD/scripts/get-helm-3 | bash
lkk@dellr530:~$ helm version
WARNING: Kubernetes configuration file is group-readable. This is insecure. Location: /home/lkk/.kube/config
WARNING: Kubernetes configuration file is world-readable. This is insecure. Location: /home/lkk/.kube/config
version.BuildInfo{Version:"v3.14.3", GitCommit:"f03cc04caaa8f6d7c3e67cf918929150cf6f3f12", GitTreeState:"clean", GoVersion:"go1.21.7"}
https://www.digitalocean.com/community/tutorials/how-to-setup-k3s-kubernetes-cluster-on-ubuntu
```

Execute the following command to see all Kubernetes objects deployed in the cluster in the kube-system namespace. kubectl is installed automatically during the K3s installation and thus does not need to be installed individually.



Installing the NVIDIA drivers on the K3s node. 

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

> A Helm Chart is a package that contains all the necessary resources to deploy an application on a Kubernetes cluster. A Helm Chart serves as a blueprint or template for deploying applications on Kubernetes. It includes all the Kubernetes resource YAML manifest files needed to run the application. In addition to Kubernetes manifests, a Helm Chart directory structure contains other files specific to Helm, such as:
Chart.yaml: Defines metadata about the chart (e.g., name, version, dependencies). values.yaml: Contains default configuration values that can be overridden during installation. Templates (Go templates): Used to render Kubernetes manifests dynamically based on user-defined values.

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