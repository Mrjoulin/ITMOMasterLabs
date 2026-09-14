data "yandex_compute_image" "ubuntu" {
  family = "ubuntu-2204-lts"
}

data "yandex_vpc_subnet" "default" {
  name = "devops-ru-central1-a"
}

resource "yandex_compute_instance" "vm" {
  name = var.vm_name

  resources {
    cores  = var.vm_cores
    memory = var.vm_memory
  }

  boot_disk {
    initialize_params {
      image_id = data.yandex_compute_image.ubuntu.id
      size     = var.vm_disk_size
    }
  }

  network_interface {
    subnet_id = data.yandex_vpc_subnet.default.id
    nat       = true
  }

  metadata = {
    ssh-keys = "ubuntu:${var.ssh_public_key}"
  }
}

output "external_ip" {
  value     = yandex_compute_instance.vm.network_interface.0.nat_ip_address
  description = "External IP address of the VM"
}

output "internal_ip" {
  value     = yandex_compute_instance.vm.network_interface.0.ip_address
  description = "Internal IP address of the VM"
}
