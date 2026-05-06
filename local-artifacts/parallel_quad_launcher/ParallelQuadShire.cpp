//******************************************************************************
// Copyright (c) 2026 AIFoundry
// SPDX-License-Identifier: Apache-2.0
//------------------------------------------------------------------------------
// Four-shire launcher using four independent runtime streams. Launch submission
// is released from four host threads at the same barrier, then completion is
// measured after all events have been queued.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "GenericLauncher.h"
#include <runtime/IRuntime.h>

struct Options {
  fs::path kernel_paths[4];
  std::string labels[4] = {"shire0", "shire1", "shire2", "shire3"};
  uint64_t shire_masks[4] = {0x1, 0x2, 0x4, 0x8};
  uint64_t loops = 100000000;
  int kernel_launch_timeout = 60;
  std::string device_type = "silicon";
  std::string launch_mode = "parallel";
};

struct ProbeArgs {
  uint64_t out_dev;
  uint64_t loops;
  uint64_t marker;
};

struct ProbeResult {
  uint64_t shire;
  uint64_t neighborhood;
  uint64_t hart;
  uint64_t minion;
  uint64_t thread;
  uint64_t loops;
  uint64_t marker;
  uint64_t accumulator;
};

static Options parse_args(int argc, char* const* argv, std::vector<char*>& nextlevel) {
  static constexpr const char* short_opts = "0:1:2:3:t:d:l:m:n:o:p:q:";
  static const std::vector<struct option> long_opts{
    {"kernel_path_0", required_argument, nullptr, '0'},
    {"kernel_path_1", required_argument, nullptr, '1'},
    {"kernel_path_2", required_argument, nullptr, '2'},
    {"kernel_path_3", required_argument, nullptr, '3'},
    {"kernel_launch_timeout", required_argument, nullptr, 't'},
    {"device_type", required_argument, nullptr, 'd'},
    {"loops", required_argument, nullptr, 'l'},
    {"launch_mode", required_argument, nullptr, 'm'},
    {"label_0", required_argument, nullptr, 'n'},
    {"label_1", required_argument, nullptr, 'o'},
    {"label_2", required_argument, nullptr, 'p'},
    {"label_3", required_argument, nullptr, 'q'},
    {nullptr, 0, nullptr, 0}};

  Options o;
  int ret;
  int idx = 0;
  opterr = 0;

  while ((ret = getopt_long_only(argc, argv, short_opts, long_opts.data(), &idx)) != -1) {
    switch (ret) {
    case '0': o.kernel_paths[0] = optarg; break;
    case '1': o.kernel_paths[1] = optarg; break;
    case '2': o.kernel_paths[2] = optarg; break;
    case '3': o.kernel_paths[3] = optarg; break;
    case 't': o.kernel_launch_timeout = atoi(optarg); break;
    case 'd': o.device_type = optarg; break;
    case 'l': o.loops = strtoull(optarg, nullptr, 0); break;
    case 'm': o.launch_mode = optarg; break;
    case 'n': o.labels[0] = optarg; break;
    case 'o': o.labels[1] = optarg; break;
    case 'p': o.labels[2] = optarg; break;
    case 'q': o.labels[3] = optarg; break;
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
  rt::IRuntime* runtime() {
    return runtime_;
  }
  const std::vector<rt::DeviceId>& devices() const {
    return devices_;
  }
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

  rt::IRuntime* rt = launcher.runtime();
  auto devices = launcher.devices();

  rt::StreamId streams[4];
  for (int i = 0; i < 4; i++) {
    streams[i] = rt->createStream(devices[0]);
  }

  std::byte* outs[4];
  uint64_t outPtrs[4];
  ProbeArgs args[4];
  for (int i = 0; i < 4; i++) {
    outs[i] = rt->mallocDevice(devices[0], sizeof(ProbeResult));
    outPtrs[i] = reinterpret_cast<uint64_t>(outs[i]);
    args[i] = ProbeArgs{outPtrs[i], opt.loops, static_cast<uint64_t>(0xabc000 + i)};
  }

  rt::EventId events[4]{};
  std::chrono::steady_clock::time_point t0;
  long submit_us[4]{};

  auto submit_one = [&](int i) {
    rt::KernelLaunchOptions opts;
    opts.setShireMask(opt.shire_masks[i]);
    opts.setBarrier(true);
    opts.setFlushL3(false);

    auto before = std::chrono::steady_clock::now();
    events[i] = rt->kernelLaunch(streams[i], kids[i], reinterpret_cast<const std::byte*>(&args[i]), sizeof(args[i]), opts);
    auto after = std::chrono::steady_clock::now();
    submit_us[i] = std::chrono::duration_cast<std::chrono::microseconds>(after - t0).count();

    printf("[submit] slot=%d label=%s mask=0x%lx event=%d submit_at=%ldus call_time=%ldus\n",
           i, opt.labels[i].c_str(), opt.shire_masks[i], int(events[i]), submit_us[i],
           std::chrono::duration_cast<std::chrono::microseconds>(after - before).count());
  };

  if (opt.launch_mode == "sequential") {
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 4; i++) {
      submit_one(i);
      bool ok = rt->waitForEvent(events[i], std::chrono::seconds(opt.kernel_launch_timeout));
      printf("[sequential] slot=%d completed ok=%d at %ldus\n", i, ok ? 1 : 0,
             std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
    }
  } else if (opt.launch_mode == "parallel") {
    std::mutex m;
    std::condition_variable cv;
    bool go = false;
    std::atomic<int> ready{0};

    auto worker = [&](int i) {
      {
        std::unique_lock<std::mutex> lock(m);
        ready.fetch_add(1);
        cv.notify_one();
        cv.wait(lock, [&] { return go; });
      }
      submit_one(i);
    };

    std::thread threads[4] = {
      std::thread(worker, 0),
      std::thread(worker, 1),
      std::thread(worker, 2),
      std::thread(worker, 3),
    };

    {
      std::unique_lock<std::mutex> lock(m);
      cv.wait(lock, [&] { return ready.load() == 4; });
      t0 = std::chrono::steady_clock::now();
      go = true;
    }
    cv.notify_all();

    for (auto& thread : threads) {
      thread.join();
    }
  } else {
    fprintf(stderr, "unsupported --launch_mode=%s, expected parallel or sequential\n", opt.launch_mode.c_str());
    return 2;
  }

  auto all_submitted = std::chrono::steady_clock::now();
  printf("[time] launch_mode=%s all 4 submit calls returned at %ldus\n",
         opt.launch_mode.c_str(), std::chrono::duration_cast<std::chrono::microseconds>(all_submitted - t0).count());

  bool wait_ok = true;
  if (opt.launch_mode == "parallel") {
    for (int i = 0; i < 4; i++) {
      wait_ok &= rt->waitForEvent(events[i], std::chrono::seconds(opt.kernel_launch_timeout));
    }
  }
  auto all_done = std::chrono::steady_clock::now();
  printf("[time] all 4 events completed at %ldus\n",
         std::chrono::duration_cast<std::chrono::microseconds>(all_done - t0).count());

  bool err = launcher.checkKernelExecutionErrors();

  for (int i = 0; i < 4; i++) {
    ProbeResult res{};
    rt->memcpyDeviceToHost(streams[i], outs[i], reinterpret_cast<std::byte*>(&res), sizeof(res));
    rt->waitForStream(streams[i], std::chrono::seconds(5));
    printf("[K%d %-8s mask=0x%lx] shire=%lu neigh=%lu hart=%lu minion=%lu thread=%lu loops=%lu marker=0x%lx acc=0x%lx\n",
           i, opt.labels[i].c_str(), opt.shire_masks[i], res.shire, res.neighborhood, res.hart, res.minion,
           res.thread, res.loops, res.marker, res.accumulator);
    rt->freeDevice(devices[0], outs[i]);
    rt->destroyStream(streams[i]);
    launcher.unLoadKernel(kids[i]);
  }

  launcher.tearDown();
  return (!wait_ok || err) ? -1 : 0;
}
