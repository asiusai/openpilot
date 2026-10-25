"""Loopback-only protocol interoperability harness. Uses disposable test identities."""
import argparse
import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from openpilot.system.asius.pqc import RootIdentity
from openpilot.system.asius.session import SecureSession


class Handler(BaseHTTPRequestHandler):
  sessions = {}

  def log_message(self, *_):
    pass

  def do_POST(self):
    try:
      length = int(self.headers.get('Content-Length', '0'))
      if not 0 < length <= 1024 * 1024:
        raise ValueError('Invalid body length')
      body = json.loads(self.rfile.read(length))
      if self.path == '/session':
        if len(self.sessions) >= 64:
          raise ValueError('Harness session limit')
        key = secrets.token_hex(16)
        session = SecureSession(RootIdentity.generate(), body['transport'])
        self.sessions[key] = session
        result = {'id': key, 'publicKey': session.public_key}
      elif self.path.startswith('/wire/'):
        session = self.sessions[self.path.removeprefix('/wire/')]
        received = session.receive(body)
        result = {'wire': received['reply']} if 'reply' in received else {'wire': session.encrypt(received['peer'], received['text'])}
      else:
        raise ValueError('Unknown harness operation')
      status = 200
    except Exception as error:
      status, result = 400, {'error': type(error).__name__}
    data = json.dumps(result).encode()
    self.send_response(status)
    self.send_header('Content-Type', 'application/json')
    self.send_header('Content-Length', str(len(data)))
    self.end_headers()
    self.wfile.write(data)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--port', type=int, default=4133)
  args = parser.parse_args()
  server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
  print(f'Protocol test server on loopback port {args.port}', flush=True)
  server.serve_forever()
