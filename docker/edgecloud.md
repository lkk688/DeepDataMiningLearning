# Edge Cloud

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