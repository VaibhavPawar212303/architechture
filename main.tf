# main.tf

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.14.0"
    }
  }
}

provider "google" {
  project = "ai-testcase" 
  region  = "us-central1"
}

resource "google_compute_network" "phi3_vpc" {
  name                    = "phi3-vision-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "phi3_subnet" {
  name          = "phi3-vision-subnet"
  ip_cidr_range = "10.0.1.0/24"
  network       = google_compute_network.phi3_vpc.id
  region        = "us-central1"
}

resource "google_compute_firewall" "allow_http_ssh" {
  name    = "allow-phi3-api-ssh"
  network = google_compute_network.phi3_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["22", "8000"]
  }

  source_ranges = ["0.0.0.0/0"]
}

resource "google_compute_instance" "phi3_vm" {
  name         = "phi3-vision-vm"
  # Use a cost-effective, general-purpose machine type
  machine_type = "e2-standard-4" # 4 vCPUs, 16 GB RAM
  zone         = "us-central1-a"

  boot_disk {
    initialize_params {
      # Switch back to Debian, as our script will install Docker
      image = "debian-cloud/debian-11"
      size  = 50 # 50 GB is enough for a CPU setup
    }
  }

  # The startup script will configure the VM
  metadata_startup_script = file("setup_vm.sh")

  network_interface {
    network    = google_compute_network.phi3_vpc.id
    subnetwork = google_compute_subnetwork.phi3_subnet.id
    access_config {} # Assigns an ephemeral public IP
  }

  service_account {
    scopes = ["cloud-platform"]
  }

  tags = ["phi3-api"]
}

output "vm_public_ip" {
  description = "The public IP address of the VM."
  value       = google_compute_instance.phi3_vm.network_interface[0].access_config[0].nat_ip
}

output "startup_script_log_command" {
  description = "Command to check the logs of the startup script."
  value       = "gcloud compute ssh ${google_compute_instance.phi3_vm.name} --zone ${google_compute_instance.phi3_vm.zone} --command 'sudo journalctl -u google-startup-scripts.service -f'"
}