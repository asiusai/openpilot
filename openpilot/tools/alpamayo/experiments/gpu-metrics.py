"""Read SMU telemetry through the existing GPU context; never alter power settings."""
def read(device):
  smu = device.iface.dev_impl.smu
  module = smu.smu_mod
  table = smu.read_table(module.SmuMetricsExternal_t, module.TABLE_SMU_METRICS).SmuMetrics
  fields = ('AverageGfxclkFrequencyTarget', 'AverageGfxclkFrequencyPreDs',
            'AverageGfxclkFrequencyPostDs', 'AverageMemclkFrequencyPreDs',
            'AverageSocketPower', 'AverageTotalBoardPower', 'AverageGfxActivity',
            'AverageUclkActivity', 'AvgFanRpm', 'PcieRate', 'PcieWidth')
  result = {key: int(getattr(table, key)) for key in fields}
  result.update(AvgTemperature=list(table.AvgTemperature), ThrottlingPercentage=list(table.ThrottlingPercentage),
                 clocks=smu.read_clocks((module.PPCLK_GFXCLK, module.PPCLK_UCLK)), smu_module=module.__name__)
  if hasattr(module, 'PPSMC_MSG_GetPptLimit'):
    result['active_power_limit_w'] = smu._send_msg(module.PPSMC_MSG_GetPptLimit, 0, read_back_arg=True)
  return result


def stock_limits(device):
  """Read firmware's board limits. Do not change the active power limit."""
  smu = device.iface.dev_impl.smu
  module = smu.smu_mod
  assert module.__name__.endswith('smu_14_0_2')
  table = smu.read_table(module.PPTable_t, module.TABLE_PPTABLE)
  return {'socket_power_ac_w': list(table.CustomSkuTable.SocketPowerLimitAc),
          'power_message_limits_w': [list(row) for row in table.SkuTable.MsgLimits.Power],
          'rest_board_power_w': int(table.BoardTable.RestBoardPower)}
