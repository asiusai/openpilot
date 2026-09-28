#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#include "system/camerad/cameras/cdm.h"

struct IspRegWrite {
  uint32_t offset;
  uint32_t value;
};

struct IspLutWrite {
  uint32_t cfg_offset;
  uint8_t bank;
  uint32_t bytes;
};

// Mainline CAMSS accepts register/LUT ioctls rather than CDM command buffers.
// Decode the same program that the downstream camera backend submits to CDM.
inline bool parse_isp_cdm(const uint8_t *data, size_t size,
                          std::vector<IspRegWrite> &regs, std::vector<IspLutWrite> &luts) {
  while (size >= sizeof(uint32_t)) {
    uint32_t word;
    std::memcpy(&word, data, sizeof(word));
    const auto opcode = word >> 24;
    const size_t count = word & 0xffff;
    size_t length;
    if (opcode == CAM_CDM_CMD_REG_CONT) {
      length = sizeof(cdm_regcontinuous_cmd) + count * sizeof(uint32_t);
      if (size < length) return false;
      cdm_regcontinuous_cmd cmd;
      std::memcpy(&cmd, data, sizeof(cmd));
      for (size_t i = 0; i < count; ++i) {
        uint32_t value;
        std::memcpy(&value, data + sizeof(cmd) + i * sizeof(value), sizeof(value));
        regs.push_back({cmd.offset + (uint32_t)i * 4, value});
      }
    } else if (opcode == CAM_CDM_CMD_REG_RANDOM) {
      length = sizeof(cdm_regrandom_cmd) + count * 2 * sizeof(uint32_t);
      if (size < length) return false;
      for (size_t i = 0; i < count; ++i) {
        IspRegWrite reg;
        std::memcpy(&reg, data + sizeof(cdm_regrandom_cmd) + i * sizeof(reg), sizeof(reg));
        regs.push_back(reg);
      }
    } else if (opcode == CAM_CDM_CMD_DMI || opcode == CAM_CDM_CMD_DMI_32 || opcode == CAM_CDM_CMD_DMI_64) {
      length = sizeof(cdm_dmi_cmd);
      if (size < length) return false;
      cdm_dmi_cmd cmd;
      std::memcpy(&cmd, data, sizeof(cmd));
      if ((cmd.length + 1) % sizeof(uint32_t) != 0) return false;
      luts.push_back({cmd.DMIAddr, (uint8_t)cmd.DMISel, cmd.length + 1U});
    } else {
      return false;
    }
    data += length;
    size -= length;
  }
  return size == 0;
}
