#include <limits>
#include "common/hardware/asius/lighting.h"
#include "common/tests/native_test.h"

void test_live_ir() {
  CHECK(live_ir_power(100, true, "", 100) == 66);
  CHECK(live_ir_power(200, false, "", 100) == 66);
  CHECK(live_ir_power(-1, false, "", 100) == 0);
  const std::string request = R"({"expires":104,"irMa":150})";
  CHECK(live_ir_power(17, false, request, 100) == 100);
  CHECK(live_ir_power(17, true, request, 100) == 11);
  CHECK(live_ir_power(17, false, request, 104) == 11);
  CHECK(live_ir_power(17, false, request, 98) == 11);
  CHECK(live_ir_power(17, false, request, std::numeric_limits<double>::quiet_NaN()) == 11);
  CHECK(live_ir_power(17, false, R"({"expires":104,"irMa":0})", 100) == 0);
  CHECK(live_ir_power(17, false, R"({"expires":104,"irMa":75})", 100) == 50);
  for (const auto &value : {"null", "true", "-1", "151", "1.5", "\"30\""}) {
    CHECK(live_ir_power(17, false, std::string("{\"expires\":104,\"irMa\":") + value + "}", 100) == 11);
  }
  for (const auto &value : {"", "invalid", "null", "[]", "{}"}) {
    CHECK(live_ir_power(17, false, value, 100) == 11);
  }
}

int main() { return run_native_test(test_live_ir); }
