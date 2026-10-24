#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include "json11/json11.hpp"

inline int live_ir_power(int automatic, bool onroad, const std::string &request, double now) {
  // Automatic exposure uses at most 100 mA of the kernel's 150 mA full scale.
  automatic = std::clamp(automatic, 0, 100) * 100 / 150;
  if (onroad || request.empty()) return automatic;
  std::string error;
  const auto state = json11::Json::parse(request, error);
  const auto &current = state["irMa"];
  const double remaining = state["expires"].number_value() - now;
  const double ma = current.number_value();
  if (!error.empty() || !current.is_number() || !std::isfinite(ma) || ma < 0 || ma > 150 || std::floor(ma) != ma ||
      remaining <= 0 || remaining > 5 || !std::isfinite(remaining)) return automatic;
  return static_cast<int>(ma * 100 / 150);
}
