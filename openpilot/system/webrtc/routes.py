"""Read recorded routes over an authenticated, encrypted WebRTC data channel."""
from __future__ import annotations

import asyncio
import contextlib
import json
import mmap
import re
import struct
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from collections.abc import Callable

from openpilot.system.loggerd.data_media import VIDEO_SAMPLE_TYPES, fragment_timing, parse_track, read_boxes, recording_boxes
from openpilot.system.loggerd.data_upload_queue import upload_status

ROUTE_RE = re.compile(r"^[A-Za-z0-9_-]+--[0-9]+$")
VIDEO_FILES = {"qcamera.mp4", "fcamera.mp4", "ecamera.mp4", "dcamera.mp4", "qcamera.ts", "fcamera.hevc", "ecamera.hevc", "dcamera.hevc"}
ROUTE_FILES = VIDEO_FILES | {"qlog", "qlog.zst", "rlog", "rlog.zst"}
MAX_RANGE_BYTES = 8 * 1024 * 1024
CHUNK_BYTES = 16 * 1024
CONVERSION_SECONDS = 6


class RecordingUnavailableError(ValueError):
  pass


def mp4_manifest(path: Path) -> dict:
  try:
    return _mp4_manifest(path)
  except (ValueError, StopIteration, struct.error, IndexError) as error:
    raise RecordingUnavailableError("This segment has no usable video frames") from error


def _mp4_manifest(path: Path) -> dict:
  with path.open("rb") as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as data:
    boxes = recording_boxes(data)
    moov = next((box for box in boxes if box.type == b"moov"), None)
    if moov is None:
      raise ValueError("recording has no initialization metadata")
    track = next((track for box in read_boxes(data, moov.payload_start, moov.end)
                  if box.type == b"trak" and (track := parse_track(data, box)) is not None and track.sample_type in VIDEO_SAMPLE_TYPES), None)
    if track is None:
      raise ValueError("recording has no supported video track")
    moofs = [box for box in boxes if box.type == b"moof"]
    if not moofs or moov.start > moofs[0].start:
      raise ValueError("recording is not a fragmented MP4")
    fragments = [{"kind": "init", "offset": 0, "length": moofs[0].start}]
    for index, box in enumerate(moofs):
      start, duration = fragment_timing(data, box, track)
      end = moofs[index + 1].start if index + 1 < len(moofs) else boxes[-1].end
      fragments.append({"kind": "media", "offset": box.start, "length": end - box.start,
                        "timeMs": round(start * 1000), "durationMs": max(1, round(duration * 1000))})
    return {"codec": track.codec, "contentType": f'video/mp4; codecs="{track.codec}"', "size": len(data),
            "recovered": boxes[-1].end < len(data), "fragments": fragments}


class RouteFiles:
  def __init__(self, root: str | Path, cache: str | Path):
    self.root = Path(root).resolve()
    self.cache = Path(cache)
    self.manifests: dict[tuple, dict] = {}
    self.media_lock = threading.RLock()

  def path(self, relative: str) -> Path:
    if not isinstance(relative, str):
      raise ValueError("invalid recording path")
    parts = relative.split("/")
    if len(parts) != 2 or not ROUTE_RE.fullmatch(parts[0]) or parts[1] not in ROUTE_FILES:
      raise ValueError("invalid recording path")
    path = self.root.joinpath(*parts)
    if path.parent.is_symlink() or path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(self.root):
      raise ValueError("recording is unavailable")
    if any(path.parent.glob("*.lock")):
      raise ValueError("recording is still being written")
    return path

  def list_routes(self) -> list[dict]:
    routes: dict[str, dict] = {}
    for directory in self.root.iterdir():
      if directory.is_symlink() or not directory.is_dir() or not ROUTE_RE.fullmatch(directory.name):
        continue
      if any(directory.glob("*.lock")):
        continue
      route_id, segment_string = directory.name.rsplit("--", 1)
      segment = int(segment_string)
      files = []
      recording_times = []
      for file in directory.iterdir():
        if file.name not in ROUTE_FILES or file.is_symlink() or not file.is_file() or Path(str(file) + ".lock").exists():
          continue
        stat = file.stat()
        if not stat.st_size:
          continue
        recording_times.append(max(0, round((stat.st_mtime - segment * 60) * 1000)))
        files.append({"path": f"{directory.name}/{file.name}", "name": file.name, "segment": segment, "size": stat.st_size,
                      **upload_status(file)})
      if not files:
        continue
      created = min(recording_times)
      route = routes.setdefault(route_id, {"routeId": route_id, "createdAt": created, "files": []})
      route["createdAt"] = min(route["createdAt"], created)
      route["files"].extend(files)
    for route in routes.values():
      route["files"].sort(key=lambda file: (file["segment"], file["name"]))
    return sorted(routes.values(), key=lambda route: route["routeId"], reverse=True)

  def media_path(self, relative: str, h264: bool = False, start: int | None = None, cancelled: threading.Event | None = None) -> Path:
    if cancelled is not None and cancelled.is_set():
      raise InterruptedError("recording request cancelled")
    if start is not None and (not h264 or type(start) is not int or start < 0 or start >= 60 or start % CONVERSION_SECONDS):
      raise ValueError("invalid video conversion range")
    path = self.path(relative)
    if path.name not in VIDEO_FILES:
      raise ValueError("file is not a video")
    if path.suffix == ".mp4" and not h264:
      return path
    # Wrap original HEVC/TS recordings without changing their encoded video.
    stat = path.stat()
    destination = self.cache / f"{path.parent.name}-{path.name}-{stat.st_size}-{stat.st_mtime_ns}-{'h264' if h264 else 'copy'}-{start}.mp4"
    if not destination.exists():
      command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
      if path.suffix == ".hevc":
        command += ["-fflags", "+genpts", "-r", "20"]
      # Input seeking jumps to the nearby keyframe, avoiding a whole-minute
      # transcode before the browser can show its first frame or seek.
      if start is not None and path.suffix != ".hevc":
        command += ["-ss", str(start)]
      command += ["-i", str(path), "-map", "0:v:0", "-an"]
      if start is not None:
        duration = CONVERSION_SECONDS
        if path.suffix == ".mp4":
          manifest = mp4_manifest(path)
          media = manifest["fragments"][1:]
          end = (media[-1]["timeMs"] + media[-1]["durationMs"] - media[0]["timeMs"]) / 1000
          duration = min(duration, end - start)
          if duration <= 0:
            raise RecordingUnavailableError("This section has no recorded frames")
        if path.suffix == ".hevc":
          command += ["-ss", str(start)]
        command += ["-t", str(duration)]
      if h264:
        command += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-threads", "2", "-pix_fmt", "yuv420p",
                    "-g", "20", "-keyint_min", "20", "-sc_threshold", "0"]
      else:
        command += ["-c:v", "copy"]
      if path.suffix == ".hevc" and not h264:
        command += ["-tag:v", "hvc1"]
      command += ["-movflags", "+frag_keyframe+empty_moov+default_base_moof+skip_sidx", str(destination)]
      try:
        with subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) as process:
          deadline = time.monotonic() + 90
          try:
            while True:
              if cancelled is not None and cancelled.is_set():
                raise InterruptedError("recording request cancelled")
              if time.monotonic() >= deadline:
                raise TimeoutError("video conversion timed out")
              try:
                _, stderr = process.communicate(timeout=0.2)
                break
              except subprocess.TimeoutExpired:
                continue
            if process.returncode:
              raise subprocess.CalledProcessError(process.returncode, command, stderr=stderr)
          except BaseException:
            process.kill()
            process.communicate()
            raise
      except subprocess.CalledProcessError as error:
        destination.unlink(missing_ok=True)
        raise RecordingUnavailableError("This segment could not be decoded") from error
      except Exception:
        destination.unlink(missing_ok=True)
        raise
      # Keep at most two prepared sections per connection.
      for old in sorted(self.cache.glob("*.mp4"), key=lambda item: item.stat().st_mtime, reverse=True)[2:]:
        old.unlink(missing_ok=True)
    return destination

  def manifest(self, relative: str, h264: bool = False, start: int | None = None, cancelled: threading.Event | None = None) -> dict:
    with self.media_lock:
      path = self.media_path(relative, h264, start, cancelled)
      stat = path.stat()
      key = (relative, h264, start, stat.st_size, stat.st_mtime_ns)
      if key not in self.manifests:
        if len(self.manifests) >= 128:
          self.manifests.clear()
        self.manifests[key] = mp4_manifest(path)
      return self.manifests[key]

  def read(self, relative: str, offset: int, length: int, media: bool = False, h264: bool = False, start: int | None = None,
           cancelled: threading.Event | None = None) -> bytes:
    if type(offset) is not int or type(length) is not int or offset < 0 or not 0 < length <= MAX_RANGE_BYTES:
      raise ValueError("invalid recording range")
    with self.media_lock:
      path = self.media_path(relative, h264, start, cancelled) if media else self.path(relative)
      if offset + length > path.stat().st_size:
        raise ValueError("recording range is outside the file")
      with path.open("rb") as source:
        source.seek(offset)
        data = source.read(length)
    if len(data) != length:
      raise ValueError("recording changed while reading")
    return data


class RouteSession:
  def __init__(self, sdp: str, root: str, authorized: Callable[[], bool]):
    from teleoprtc.builder import WebRTCAnswerBuilder
    self.identifier = str(uuid.uuid4())
    self.stream = WebRTCAnswerBuilder(sdp).stream()
    self.authorized = authorized
    cache_root = Path(root).parent / ".route-playback"
    cache_root.mkdir(mode=0o700, exist_ok=True)
    self.temporary = tempfile.TemporaryDirectory(prefix="session-", dir=cache_root)
    self.files = RouteFiles(root, self.temporary.name)
    self.requests: dict[int, asyncio.Task] = {}
    self.last_activity = time.monotonic()
    self.run_task: asyncio.Task | None = None
    self.closed = False
    self.send_lock = asyncio.Lock()
    self.drained = asyncio.Event()
    self.channel_ready = False
    self.stream.set_message_handler(self.message_handler)

  async def get_answer(self):
    return await self.stream.start()

  def start(self):
    self.run_task = asyncio.create_task(self.run())

  def message_handler(self, message: bytes | str):
    try:
      if len(message) > 8192:
        raise ValueError("request too large")
      request = json.loads(message)
      identifier = request.get("id")
      if type(identifier) is not int or not 0 < identifier < 2**32:
        raise ValueError("invalid request id")
      if request.get("op") == "cancel":
        if task := self.requests.get(identifier):
          task.cancel()
        return
      if identifier in self.requests or len(self.requests) >= 4:
        raise ValueError("too many requests")
      self.last_activity = time.monotonic()
      task = asyncio.create_task(self.respond(identifier, request))
      self.requests[identifier] = task
      task.add_done_callback(lambda _: self.requests.pop(identifier, None))
    except (ValueError, TypeError, AttributeError):
      pass

  async def send(self, message: bytes | str):
    channel = self.stream.get_messaging_channel()
    if not self.channel_ready:
      loop = asyncio.get_running_loop()
      channel.set_buffered_amount_low_threshold(0)
      channel.on_buffered_amount_low(lambda: loop.call_soon_threadsafe(self.drained.set))
      self.channel_ready = True
    async with self.send_lock:
      if self.closed or not channel.is_open():
        raise ConnectionError("route connection closed")
      self.drained.clear()
      # send() returns false when SCTP queues the message. Use its drain callback:
      # libdatachannel-py 2026.1.0.dev2's inherited buffered_amount() binding crashes.
      if not channel.send(message):
        await asyncio.wait_for(self.drained.wait(), timeout=20)

  async def respond(self, identifier: int, request: dict):
    cancelled = threading.Event()
    try:
      if not self.authorized():
        raise PermissionError("device access has been removed")
      match request.get("op"):
        case "list":
          result = await asyncio.to_thread(self.files.list_routes)
        case "manifest":
          result = await asyncio.to_thread(self.files.manifest, request.get("path"), request.get("h264") is True, request.get("start"), cancelled)
        case "read":
          result = await asyncio.to_thread(self.files.read, request.get("path"), request.get("offset"), request.get("length"),
                                           request.get("media") is True, request.get("h264") is True, request.get("start"), cancelled)
        case "ping":
          result = True
        case _:
          raise ValueError("unknown recording request")
      # Both JSON metadata and file bytes are chunked below the SCTP message limit.
      binary = isinstance(result, bytes)
      data = result if binary else json.dumps(result).encode()
      await self.send(json.dumps({"id": identifier, "length": len(data), "binary": binary}))
      header = struct.pack(">I", identifier)
      for start in range(0, len(data), CHUNK_BYTES):
        await self.send(header + data[start:start + CHUNK_BYTES])
      await self.send(json.dumps({"id": identifier, "done": True}))
    except asyncio.CancelledError:
      cancelled.set()
      raise
    except Exception as error:
      with contextlib.suppress(Exception):
        await self.send(json.dumps({"id": identifier, "error": str(error),
                                    **({"code": "recording_unavailable"} if isinstance(error, RecordingUnavailableError) else {})}))

  async def run(self):
    try:
      await asyncio.wait_for(self.stream.wait_for_connection(), timeout=30)
      disconnected = asyncio.create_task(self.stream.wait_for_disconnection())
      try:
        while not disconnected.done() and self.authorized() and time.monotonic() - self.last_activity < 120:
          await asyncio.wait({disconnected}, timeout=10)
      finally:
        disconnected.cancel()
        with contextlib.suppress(asyncio.CancelledError):
          await disconnected
    finally:
      await self.stop()

  async def stop(self):
    if self.closed:
      return
    self.closed = True
    if self.run_task and self.run_task is not asyncio.current_task():
      self.run_task.cancel()
      with contextlib.suppress(asyncio.CancelledError):
        await self.run_task
    tasks = list(self.requests.values())
    for task in tasks:
      task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await self.stream.stop()
    self.temporary.cleanup()
