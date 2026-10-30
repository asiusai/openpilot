def start_pairing() -> str:
  # Keep BlueZ dependencies out of the UI startup path on stock AGNOS.
  try:
    from openpilot.system.asius.bluetoothd import enable_pairing_mode
    enable_pairing_mode()
  except ImportError:
    return "Bluetooth setup is incomplete. Install the Bluetooth dependencies first."
  except PermissionError:
    return "Turn the vehicle off before Bluetooth pairing."
  return "Bluetooth pairing is on for two minutes. Select this device in Asius App. Compare the six digits before confirming."


class PairingDisplay:
  """Local touchscreen approval for comma devices. No code is sent to the app."""
  def __init__(self):
    self.request_id = None
    self.widget = None
    self.answered = False

  def tick(self):
    from openpilot.system.asius.pairing_confirmation import read_confirmation, send_ui_response
    from openpilot.system.ui.lib.application import gui_app
    from openpilot.system.ui.widgets import DialogResult
    from openpilot.selfdrive.ui.ui_state import ui_state

    status = None if ui_state.started else read_confirmation()
    current = status['id'] if status else None

    def answer(request_id, accepted):
      if request_id == self.request_id:
        self.answered = True
      try:
        send_ui_response(request_id, accepted)
      except OSError:
        pass  # The request can expire while the dialog is closing.

    if self.widget and not gui_app.widget_in_stack(self.widget):
      if not self.answered:
        answer(self.request_id, False)
      self.widget = None
    if current == self.request_id:
      return
    if self.widget:
      if gui_app.get_active_widget() is not self.widget:
        return
      gui_app.pop_widget()
      self.widget = None
    self.request_id, self.answered = current, False
    if status is None:
      return
    digits = status['digits']
    number = f'{digits[:3]} {digits[3:]}'
    if gui_app.big_ui():
      from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
      self.widget = ConfirmDialog(f'Bluetooth: {number}\nDoes this match your phone?\nConfirm within 30 seconds.', 'Matches',
                                  callback=lambda result, request_id=current: answer(request_id, result == DialogResult.CONFIRM))
    else:
      from openpilot.selfdrive.ui.mici.widgets.dialog import BigConfirmationDialog
      self.widget = BigConfirmationDialog(f'{number}\nmatches phone?', gui_app.texture('icons_mici/settings/device/phone.png', 85, 64),
                                          confirm_callback=lambda request_id=current: answer(request_id, True))
    gui_app.push_widget(self.widget)
