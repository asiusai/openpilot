#!/usr/bin/env python3

from openpilot.system.athena.manage_athenad import manage


def main() -> None:
  manage("openpilot.system.asius.relayd", "relayd", "RelayPid")


if __name__ == "__main__":
  main()
