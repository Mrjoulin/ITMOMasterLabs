variable "yc_iam_token" {
  type        = string
  description = "IAM token for Yandex Cloud"
  sensitive   = true
}

variable "yc_cloud_id" {
  type        = string
  description = "Yandex Cloud ID"
}

variable "yc_folder_id" {
  type        = string
  description = "Folder ID in Yandex Cloud"
}

variable "yc_zone" {
  type        = string
  description = "Default availability zone"
  default     = "ru-central1-a"
}

variable "vm_name" {
  type        = string
  description = "Virtual machine name"
  default     = "terraform-vm"
}

variable "vm_cores" {
  type        = number
  description = "Number of CPU cores"
  default     = 2
}

variable "vm_memory" {
  type        = number
  description = "RAM size in GB"
  default     = 2
}

variable "vm_disk_size" {
  type        = number
  description = "Boot disk size in GB"
  default     = 32
}

variable "ssh_public_key" {
  type        = string
  description = "SSH public key for VM access"
}
