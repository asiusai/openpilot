import unittest

import numpy as np

from openpilot.cereal import log
from openpilot.selfdrive.alpamayo.planner import PERIOD, T, PlanSelector, packet


class HandoffTest(unittest.TestCase):
  def setUp(self):
    self.sel = PlanSelector()

  def receive(self, n, **changes):
    q = n*PERIOD
    p = dict(queryMonoTime=q, readyMonoTime=q+400_000_000, slotDeadline=q+PERIOD,
             x=(T[1:]*8).tolist(), y=np.zeros(64).tolist(), yaw=np.zeros(64).tolist())
    e = log.Event.new_message(alpamayoPlan=dict(p, **changes), valid=True)
    with log.Event.from_bytes(e.to_bytes()) as event:
      self.sel.receive(event.alpamayoPlan, q+450_000_000)
    return q+450_000_000

  def output(self, now, shift=0., stop=False):
    action = dict(desiredAcceleration=-.1, desiredCurvature=.002, shouldStop=stop)
    m = log.ModelDataV2.new_message(position=dict(t=T.tolist(), x=(T*8).tolist(), y=(T*shift).tolist()),
                                    action=action, localAction=action)
    self.sel.apply(m, now, 8, .5, .5)
    self.assertAlmostEqual(m.localAction.desiredCurvature, .002)
    if not m.alpamayoQueryMonoTime:
      self.assertEqual(m.action.to_dict(), m.localAction.to_dict())
    else:
      self.assertAlmostEqual(m.action.desiredCurvature, 0)
    return bool(m.alpamayoQueryMonoTime)

  def test_select_drop_recover_and_expire(self):
    self.assertFalse(self.output(self.receive(20)))
    self.assertTrue(self.output(self.receive(21)))
    self.sel.reject()  # Supervisor reports USB loss or a skipped slot.
    self.assertFalse(self.output(22*PERIOD))
    self.assertFalse(self.output(self.receive(23)))
    now = self.receive(24)
    self.assertTrue(self.output(now))
    self.assertFalse(self.output(now, stop=True))
    self.assertFalse(self.output(now, shift=1.))
    self.assertFalse(self.output(26*PERIOD))  # Worker disappears without a final message.
    self.assertFalse(self.output(self.receive(28)))
    self.assertTrue(self.output(self.receive(29)))

  def test_bad_packets_and_startup(self):
    self.receive(10)
    self.assertTrue(self.output(self.receive(11)))
    for idx, change in enumerate((dict(x=[0.]*63), dict(yaw=[float('nan')]*64), dict(slotDeadline=0)), 12):
      self.assertFalse(self.output(self.receive(idx, **change)))
    for startup, met in ((True, True), (False, False)):
      self.assertIsNone(packet(dict(startup=startup, period_ms=500, slot_deadline_met=met)))
