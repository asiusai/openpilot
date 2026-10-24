#include <limits>
#include "common/hardware/asius/lighting.h"
#include "common/tests/native_test.h"

void test_live_ir() {
  const std::string request = R"({"expires":104,"irMa":30})";
  CHECK(live_ir_power(17, false, request, 100) == 100);
  CHECK(live_ir_power(17, true, request, 100) == 17);
  CHECK(live_ir_power(17, false, request, 104) == 17);
  CHECK(live_ir_power(17, false, request, 98) == 17);
  CHECK(live_ir_power(17, false, request, std::numeric_limits<double>::quiet_NaN()) == 17);
  CHECK(live_ir_power(17, false, R"({"expires":104,"irMa":0})", 100) == 0);
  CHECK(live_ir_power(17, false, R"({"expires":104,"irMa":15})", 100) == 50);
  for (const auto &value : {"null", "true", "-1", "31", "1.5", "\"30\""}) {
    CHECK(live_ir_power(17, false, std::string("{\"expires\":104,\"irMa\":") + value + "}", 100) == 17);
  }
  for (const auto &value : {"", "invalid", "null", "[]", "{}"}) {
    CHECK(live_ir_power(17, false, value, 100) == 17);
  }
}

int main() { return run_native_test(test_live_ir); }
