"""Hostname form of the user-facing DeviceName parameter."""
import re
import socket
import subprocess


def device_hostname(name: str) -> str:
  # Match the boot script's LC_ALL=C normalization, including non-ASCII names.
  ascii_name = name.encode('ascii', errors='replace').decode().lower()
  return re.sub(r'[^a-z0-9]+', '-', ascii_name).strip('-')[:63].rstrip('-') or 'asius-v0'


def sync_device_hostname(name: str) -> None:
  hostname = device_hostname(name)
  if socket.gethostname() != hostname:
    subprocess.run(['sudo', '-n', 'hostname', hostname], check=True, timeout=5)
