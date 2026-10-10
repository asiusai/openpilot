import asyncio
import base64
from collections import deque
import json
import logging
from typing import Any

import capnp

from openpilot.cereal import messaging


class CerealOutgoingMessageProxy:
  def __init__(self, services: list[str], enabled: bool = True, raw: bool = False):
    self.logger = logging.getLogger("webrtcd")
    self.services = list(services)
    self.raw = raw
    self.can_sock = None
    if raw:
      self.socks = [messaging.sub_sock(s) for s in self.services]
    else:
      # CAN is an event stream. Preserve all packets for the app's JSON bridge.
      self.can_sock = messaging.sub_sock("can", conflate=False, timeout=0) if "can" in services else None
      self.sm = messaging.SubMaster([s for s in services if s != "can"], frequency=100)
    self.can_sequence = 0
    self.can_pending: deque[tuple[int, int]] = deque()
    self.can_pending_bytes = 0
    self.channels = []
    self._enabled = enabled

  def add_channel(self, channel):
    self.channels.append(channel)

  def enable(self, enable: bool):
    self._enabled = enable

  def to_json(self, msg_content: Any):
    if isinstance(msg_content, capnp._DynamicStructReader):
      msg_dict = msg_content.to_dict()
    elif isinstance(msg_content, capnp._DynamicListReader):
      msg_dict = [self.to_json(msg) for msg in msg_content]
    elif isinstance(msg_content, bytes):
      msg_dict = msg_content.decode()
    else:
      msg_dict = msg_content

    return msg_dict

  def update_raw(self):
    for sock in self.socks:
      for dat in messaging.drain_sock_raw(sock):
        for channel in self.channels:
          if channel.is_open() and len(dat) <= channel.max_message_size():
            channel.send(dat)

  def acknowledge_can(self, sequence: int):
    if type(sequence) is not int or not 0 < sequence <= self.can_sequence:
      return
    while self.can_pending and self.can_pending[0][0] <= sequence:
      self.can_pending_bytes -= self.can_pending.popleft()[1]

  def update(self):
    if self.raw:
      self.update_raw()
      return

    if self.can_sock is not None:
      # Acknowledgements bound the queue even when the browser stops rendering.
      # libdatachannel-py buffered_amount() crashes on the pinned native binding.
      for _ in range(256):
        if self.can_pending_bytes > 1024 * 1024:
          for channel in self.channels:
            if channel.is_open():
              channel.close()
          self._enabled = False
          return
        event = messaging.recv_one_or_none(self.can_sock)
        if event is None:
          break
        self.can_sequence += 1
        encoded = json.dumps({
          "type": "can", "sequence": self.can_sequence, "logMonoTime": str(event.logMonoTime),
          "valid": event.valid, "encoding": "base64",
          "data": [{"address": m.address, "src": m.src, "dat": base64.b64encode(m.dat).decode("ascii")} for m in event.can],
        }).encode()
        self.can_pending.append((self.can_sequence, len(encoded)))
        self.can_pending_bytes += len(encoded)
        for channel in self.channels:
          if channel.is_open():
            channel.send(encoded)
    # this is blocking in async context...
    self.sm.update(0)
    for service, updated in self.sm.updated.items():
      if not updated:
        continue
      msg_dict = self.to_json(self.sm[service])
      mono_time, valid = self.sm.logMonoTime[service], self.sm.valid[service]
      outgoing_msg = {"type": service, "logMonoTime": mono_time, "valid": valid, "data": msg_dict}
      encoded_msg = json.dumps(outgoing_msg).encode()
      for channel in self.channels:
        if not channel.is_open():
          continue
        channel.send(encoded_msg)

  async def run(self):
    while True:
      if not self._enabled:
        await asyncio.sleep(0.01)
        continue
      try:
        self.update()
      except Exception:
        self.logger.exception("Cereal outgoing proxy failure")
      await asyncio.sleep(0.01)


class CerealIncomingMessageProxy:
  def __init__(self, pm: messaging.PubMaster):
    self.pm = pm

  def send(self, message: bytes):
    msg_json = json.loads(message)
    msg_type, msg_data = msg_json["type"], msg_json["data"]
    size = None
    if not isinstance(msg_data, dict):
      size = len(msg_data)

    msg = messaging.new_message(msg_type, size=size)
    setattr(msg, msg_type, msg_data)
    self.pm.send(msg_type, msg)


class DynamicPubMaster(messaging.PubMaster):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.lock = asyncio.Lock()

  async def add_services_if_needed(self, services):
    async with self.lock:
      for service in services:
        if service not in self.sock:
          self.sock[service] = messaging.pub_sock(service)
