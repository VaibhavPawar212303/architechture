terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.14.0"
    }
  }
  required_version = ">= 1.0"
}

provider "google" {
  project = "ai-testcase"
  region  = "us-central1"
}

# Variables for better configuration management
variable "project_id" {
  description = "The GCP project ID"
  type        = string
  default     = "ai-testcase"
}

variable "region" {
  description = "The GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "The GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "allowed_ssh_ips" {
  description = "List of IP addresses allowed to SSH"
  type        = list(string)
  default     = ["0.0.0.0/0"] # Change this to your IP for better security
}

# VPC Network
resource "google_compute_network" "phi3_vpc" {
  name                    = "phi3-vision-vpc"
  auto_create_subnetworks = false
  description             = "VPC for Phi-3 Vision API"
}

# Subnet
resource "google_compute_subnetwork" "phi3_subnet" {
  name          = "phi3-vision-subnet"
  ip_cidr_range = "10.0.1.0/24"
  network       = google_compute_network.phi3_vpc.id
  region        = var.region
  description   = "Subnet for Phi-3 Vision API"
}

# Service Account for the VM
resource "google_service_account" "phi3_sa" {
  account_id   = "phi3-vision-sa"
  display_name = "Phi-3 Vision Service Account"
  description  = "Service account for Phi-3 Vision API VM"
}

# IAM binding for the service account (minimal permissions)
resource "google_project_iam_member" "phi3_sa_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.phi3_sa.email}"
}

resource "google_project_iam_member" "phi3_sa_monitoring" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.phi3_sa.email}"
}

# Firewall rule for SSH access
resource "google_compute_firewall" "allow_ssh" {
  name    = "allow-phi3-ssh"
  network = google_compute_network.phi3_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = var.allowed_ssh_ips
  target_tags   = ["phi3-api"]
  description   = "Allow SSH access to Phi-3 Vision API VM"
}

# Firewall rule for API access
resource "google_compute_firewall" "allow_api" {
  name    = "allow-phi3-api"
  network = google_compute_network.phi3_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["8000"]
  }

  source_ranges = ["0.0.0.0/0"] # Consider restricting this for production
  target_tags   = ["phi3-api"]
  description   = "Allow API access to Phi-3 Vision API"
}

# Firewall rule for health checks
resource "google_compute_firewall" "allow_health_check" {
  name    = "allow-phi3-health-check"
  network = google_compute_network.phi3_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["8000"]
  }

  source_ranges = ["130.211.0.0/22", "35.191.0.0/16"] # Google Cloud health check ranges
  target_tags   = ["phi3-api"]
  description   = "Allow health checks for load balancer"
}

# VM Instance
resource "google_compute_instance" "phi3_vm" {
  name         = "phi3-vision-vm"
  machine_type = "e2-standard-4" # 4 vCPUs, 16 GB RAM
  zone         = var.zone

  # For GPU support, uncomment the following and adjust machine type
  # machine_type = "n1-standard-4"
  # guest_accelerator {
  #   type  = "nvidia-tesla-t4"
  #   count = 1
  # }
  # scheduling {
  #   on_host_maintenance = "TERMINATE"
  # }

  boot_disk {
    initialize_params {
      # For GPU support, use: "deeplearning-platform-release/pytorch-latest-gpu"
      image = "debian-cloud/debian-11"
      size  = 50 # GB
      type  = "pd-standard"
    }
    auto_delete = true
  }

  # Startup script
  metadata_startup_script = file("setup_vm.sh")

  metadata = {
    enable-oslogin = "TRUE"
  }

  network_interface {
    network    = google_compute_network.phi3_vpc.id
    subnetwork = google_compute_subnetwork.phi3_subnet.id
    access_config {
      # Ephemeral public IP
    }
  }

  service_account {
    email  = google_service_account.phi3_sa.email
    scopes = [
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring.write",
      "https://www.googleapis.com/auth/devstorage.read_only"
    ]
  }

  tags = ["phi3-api"]

  labels = {
    environment = "development"
    application = "phi3-vision-api"
  }

  # Allow stopping for updates
  allow_stopping_for_update = true
}

# Health Check for Load Balancer (optional)
resource "google_compute_health_check" "phi3_health_check" {
  name = "phi3-vision-health-check"

  http_health_check {
    port         = 8000
    request_path = "/health"
  }

  check_interval_sec  = 30
  timeout_sec         = 10
  healthy_threshold   = 2
  unhealthy_threshold = 3
}

# Outputs
output "vm_public_ip" {
  description = "The public IP address of the VM"
  value       = google_compute_instance.phi3_vm.network_interface[0].access_config[0].nat_ip
}

output "vm_private_ip" {
  description = "The private IP address of the VM"
  value       = google_compute_instance.phi3_vm.network_interface[0].network_ip
}

output "api_url" {
  description = "The API URL"
  value       = "http://${google_compute_instance.phi3_vm.network_interface[0].access_config[0].nat_ip}:8000"
}

output "health_check_url" {
  description = "The health check URL"
  value       = "http://${google_compute_instance.phi3_vm.network_interface[0].access_config[0].nat_ip}:8000/health"
}

output "ssh_command" {
  description = "SSH command to connect to the VM"
  value       = "gcloud compute ssh ${google_compute_instance.phi3_vm.name} --zone ${google_compute_instance.phi3_vm.zone}"
}

output "startup_script_log_command" {
  description = "Command to check the logs of the startup script"
  value       = "gcloud compute ssh ${google_compute_instance.phi3_vm.name} --zone ${google_compute_instance.phi3_vm.zone} --command 'sudo journalctl -u google-startup-scripts.service -f'"
}

output "container_logs_command" {
  description = "Command to check the container logs"
  value       = "gcloud compute ssh ${google_compute_instance.phi3_vm.name} --zone ${google_compute_instance.phi3_vm.zone} --command 'sudo docker logs phi3-container -f'"
}