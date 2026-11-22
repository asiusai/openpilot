#include "system/camerad/cameras/camera_common.h"

#include <cassert>
#include <cstdlib>
#include <cstring>

#include "common/params.h"
#include "common/util.h"

int main(int argc, char *argv[]) {
  // doesn't need RT priority since we're using isolcpus
  int ret = util::set_core_affinity({6});
  assert(ret == 0 || Params().getBool("IsOffroad")); // failure ok while offroad due to offlining cores

  if (argc > 1 && strcmp(argv[1], "--no-cabin") == 0) setenv("CAMERAD_NO_CABIN", "1", 1);
  camerad_thread();
  return 0;
}
