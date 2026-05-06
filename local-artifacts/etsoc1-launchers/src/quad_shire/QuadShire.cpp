//******************************************************************************
// Copyright (c) 2026 AIFoundry
// SPDX-License-Identifier: Apache-2.0
//------------------------------------------------------------------------------
// Four-shire concurrent launcher: run a different ELF on each of 4 shires.
// Loads up to four kernels and fires kernelLaunch back-to-back with disjoint
// shire masks before any waitKernelCompletion.

#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <getopt.h>
#include <string>
#include <vector>
#include <chrono>

#include "GenericLauncher.h"
#include <runtime/IRuntime.h>

struct Options {
  fs::path kernel_paths[4];
  std::string labels[4]      = {"iris-tree", "iris-rf", "wine-rf", "iris-tree-2"};
  uint64_t shire_masks[4]    = {0x1, 0x2, 0x4, 0x8};
  int      kernel_launch_timeout = 60;
  std::string device_type    = "silicon";
};

static Options parse_args(int argc, char* const* argv, std::vector<char*>& nextlevel) {
  static constexpr const char* short_opts = "0:1:2:3:t:d:n:o:p:q:";
  static const std::vector<struct option> long_opts{
    {"kernel_path_0",         required_argument, nullptr, '0'},
    {"kernel_path_1",         required_argument, nullptr, '1'},
    {"kernel_path_2",         required_argument, nullptr, '2'},
    {"kernel_path_3",         required_argument, nullptr, '3'},
    {"label_0",               required_argument, nullptr, 'n'},
    {"label_1",               required_argument, nullptr, 'o'},
    {"label_2",               required_argument, nullptr, 'p'},
    {"label_3",               required_argument, nullptr, 'q'},
    {"kernel_launch_timeout", required_argument, nullptr, 't'},
    {"device_type",           required_argument, nullptr, 'd'},
    {nullptr, 0, nullptr, 0}};
  Options o; int ret, idx = 0; opterr = 0;
  while ((ret = getopt_long_only(argc, argv, short_opts, long_opts.data(), &idx)) != -1) {
    switch (ret) {
    case '0': o.kernel_paths[0] = optarg; break;
    case '1': o.kernel_paths[1] = optarg; break;
    case '2': o.kernel_paths[2] = optarg; break;
    case '3': o.kernel_paths[3] = optarg; break;
    case 'n': o.labels[0] = optarg; break;
    case 'o': o.labels[1] = optarg; break;
    case 'p': o.labels[2] = optarg; break;
    case 'q': o.labels[3] = optarg; break;
    case 't': o.kernel_launch_timeout = atoi(optarg); break;
    case 'd': o.device_type = optarg; break;
    case '?': nextlevel.emplace_back(argv[optind - 1]); break;
    default: exit(1);
    }
  }
  return o;
}

class Launcher : public GenericLauncher {
public:
  Launcher() = delete;
  using GenericLauncher::GenericLauncher;
};

int main(int argc, char** argv) {
  std::vector<char*> argvPending{argv[0]};
  Options opt = parse_args(argc, argv, argvPending);

  Config config{modeFromString(opt.device_type), 1};
  config.dump();

  Launcher launcher(config, static_cast<int>(argvPending.size()), argvPending.data());
  launcher.initialize();

  rt::KernelId kids[4];
  for (int i = 0; i < 4; i++) {
    kids[i] = launcher.loadKernel(opt.kernel_paths[i]);
    std::cout << "loadKernel slot=" << i << " (" << opt.labels[i] << ") "
              << opt.kernel_paths[i] << " id=" << int(kids[i]) << "\n";
  }

  rt::IRuntime* rt = launcher.getRuntime();
  auto devices = rt->getDevices();
  static constexpr size_t kOutBytes = 15 * sizeof(int32_t);

  std::byte* outs[4];
  uint64_t   outPtrs[4];
  for (int i = 0; i < 4; i++) {
    outs[i] = rt->mallocDevice(devices[0], kOutBytes);
    outPtrs[i] = reinterpret_cast<uint64_t>(outs[i]);
  }

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < 4; i++) {
    auto t = std::chrono::steady_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t - t0).count();
    printf("[time t=%ldms] kernelLaunch %d (%s) on shire_mask=0x%lx\n",
           dt, i, opt.labels[i].c_str(), opt.shire_masks[i]);
    launcher.kernelLaunch(kids[i], &outPtrs[i], nullptr, 0, 0, opt.shire_masks[i]);
  }
  auto t_fired = std::chrono::steady_clock::now();
  auto dt_fired = std::chrono::duration_cast<std::chrono::milliseconds>(t_fired - t0).count();
  printf("[time t=%ldms] all 4 launched, waiting...\n", dt_fired);

  launcher.waitKernelCompletion(std::chrono::seconds(opt.kernel_launch_timeout));
  auto t_done = std::chrono::steady_clock::now();
  auto dt_done = std::chrono::duration_cast<std::chrono::milliseconds>(t_done - t0).count();
  printf("[time t=%ldms] all 4 done\n", dt_done);

  bool err = launcher.checkKernelExecutionErrors();

  for (int i = 0; i < 4; i++) {
    int32_t res[15] = {0};
    rt::StreamId s = rt->createStream(devices[0]);
    rt->memcpyDeviceToHost(s, outs[i], reinterpret_cast<std::byte*>(res), kOutBytes);
    rt->waitForStream(s, std::chrono::seconds(5));
    rt->destroyStream(s);
    printf("[K%d %-12s shire=0x%lx] preds:", i, opt.labels[i].c_str(), opt.shire_masks[i]);
    for (int j = 0; j < 5; ++j) printf(" [%d]=%d", j, res[j]);
    printf("\n");
    rt->freeDevice(devices[0], outs[i]);
    launcher.unLoadKernel(kids[i]);
  }
  launcher.tearDown();
  return err ? -1 : 0;
}
