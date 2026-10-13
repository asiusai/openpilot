def start_pairing() -> str:
  # Keep BlueZ dependencies out of the UI startup path on stock AGNOS.
  try:
    from openpilot.system.asius.bluetoothd import enable_pairing_mode
    enable_pairing_mode()
  except ImportError:
    return "Bluetooth setup is incomplete. Install the Bluetooth dependencies first."
  except PermissionError:
    return "Turn the vehicle off before Bluetooth pairing."
  return "Bluetooth pairing is on for five minutes. Select this device in Asius App."
