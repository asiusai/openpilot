
from openpilot.system.athena.manage_athenad import manage


def main() -> None:
  manage("openpilot.system.asius.relayd", "relayd", "RelaydPid")


if __name__ == "__main__":
  main()
