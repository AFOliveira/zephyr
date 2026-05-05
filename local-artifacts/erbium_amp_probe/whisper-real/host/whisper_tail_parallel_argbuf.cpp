//******************************************************************************
// Copyright (c) 2026 AIFoundry
// SPDX-License-Identifier: Apache-2.0
//------------------------------------------------------------------------------
// Launch multiple Whisper decoder-tail vocab tiles from one host runtime.
//
// This is intentionally narrower than a generic soc1sim launcher. It stages the
// fixed tail-argbuf layout used by whisper_decoder_tail_ln_logits_argbuf.c and
// launches each tile on a separate shire through its own runtime stream.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <device-layer/IDeviceLayer.h>
#include <runtime/IRuntime.h>
#include <runtime/Types.h>

namespace {

constexpr size_t kErbiumMemSize = 16 * 1024 * 1024;
constexpr size_t kCompactDumpSize = 0x2000;

constexpr uint64_t kZeroOffset = 0x0;
constexpr uint64_t kLnInOffset = 0x10000;
constexpr uint64_t kLnWeightOffset = 0x20000;
constexpr uint64_t kLnBiasOffset = 0x30000;
constexpr uint64_t kLnRefOffset = 0x40000;
constexpr uint64_t kWeightOffset = 0x50000;

struct TileSpec {
  uint32_t shire = 0;
  std::string elf;
  std::string weight;
  std::string dump;
  std::string log;
};

struct Options {
  std::string zero;
  std::string act;
  std::string ln_weight;
  std::string ln_bias;
  std::string ln_ref;
  uint64_t timeout_secs = 120;
  std::vector<TileSpec> tiles;
};

struct TileRuntime {
  TileSpec spec;
  rt::StreamId stream{};
  std::byte *device_buf = nullptr;
  uint64_t device_buf_arg = 0;
  rt::LoadCodeResult load{};
  bool ok = true;
  double wait_s = 0.0;
  std::vector<std::string> messages;
};

[[noreturn]] void die(const std::string &msg)
{
  std::cerr << "Error: " << msg << "\n";
  std::exit(1);
}

std::vector<std::string> split(const std::string &s, char sep)
{
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;

  while (std::getline(ss, item, sep)) {
    out.push_back(item);
  }

  return out;
}

std::vector<std::byte> read_file(const std::string &path)
{
  const auto size = std::filesystem::file_size(path);
  std::vector<std::byte> buf(size);
  std::ifstream f(path, std::ios::binary);

  if (!f) {
    die("cannot open '" + path + "'");
  }
  f.read(reinterpret_cast<char *>(buf.data()),
         static_cast<std::streamsize>(size));
  return buf;
}

void write_file(const std::string &path, const std::vector<std::byte> &buf)
{
  std::ofstream f(path, std::ios::binary);

  if (!f) {
    die("cannot open output '" + path + "'");
  }
  f.write(reinterpret_cast<const char *>(buf.data()),
          static_cast<std::streamsize>(buf.size()));
}

void write_log(const TileRuntime &tile)
{
  std::ofstream f(tile.spec.log);

  if (!f) {
    std::cerr << "Warning: cannot write log '" << tile.spec.log << "'\n";
    return;
  }

  f << "Parallel tail launcher shire: " << tile.spec.shire << "\n";
  f << "Kernel wait seconds: " << tile.wait_s << "\n";
  for (const auto &msg : tile.messages) {
    f << msg << "\n";
  }
  f << (tile.ok ? "Kernel completed successfully\n" : "Kernel failed\n");
}

void print_usage(const char *prog)
{
  std::cerr
      << "Usage: " << prog << " [options]\n\n"
      << "Required common inputs:\n"
      << "  --zero <file>\n"
      << "  --act <file>\n"
      << "  --ln-weight <file>\n"
      << "  --ln-bias <file>\n"
      << "  --ln-ref <file>\n"
      << "  --tile <shire>,<elf>,<weight>,<dump>,<log>  (repeat)\n\n"
      << "Optional:\n"
      << "  --timeout <seconds>  default 120\n";
}

TileSpec parse_tile(const std::string &arg)
{
  const auto parts = split(arg, ',');

  if (parts.size() != 5) {
    die("--tile expects <shire>,<elf>,<weight>,<dump>,<log>");
  }

  TileSpec tile;
  tile.shire = static_cast<uint32_t>(std::stoul(parts[0], nullptr, 0));
  tile.elf = parts[1];
  tile.weight = parts[2];
  tile.dump = parts[3];
  tile.log = parts[4];
  return tile;
}

Options parse_args(int argc, char **argv)
{
  static const struct option long_opts[] = {
      {"zero", required_argument, nullptr, 'z'},
      {"act", required_argument, nullptr, 'a'},
      {"ln-weight", required_argument, nullptr, 'w'},
      {"ln-bias", required_argument, nullptr, 'b'},
      {"ln-ref", required_argument, nullptr, 'r'},
      {"tile", required_argument, nullptr, 'T'},
      {"timeout", required_argument, nullptr, 't'},
      {"help", no_argument, nullptr, 'h'},
      {nullptr, 0, nullptr, 0},
  };

  Options opts;
  int c;
  while ((c = getopt_long(argc, argv, "h", long_opts, nullptr)) != -1) {
    switch (c) {
    case 'z':
      opts.zero = optarg;
      break;
    case 'a':
      opts.act = optarg;
      break;
    case 'w':
      opts.ln_weight = optarg;
      break;
    case 'b':
      opts.ln_bias = optarg;
      break;
    case 'r':
      opts.ln_ref = optarg;
      break;
    case 'T':
      opts.tiles.push_back(parse_tile(optarg));
      break;
    case 't':
      opts.timeout_secs = std::stoull(optarg, nullptr, 0);
      break;
    case 'h':
      print_usage(argv[0]);
      std::exit(0);
    default:
      print_usage(argv[0]);
      std::exit(1);
    }
  }

  if (opts.zero.empty() || opts.act.empty() || opts.ln_weight.empty() ||
      opts.ln_bias.empty() || opts.ln_ref.empty()) {
    die("all common input files are required");
  }
  if (opts.tiles.empty()) {
    die("at least one --tile is required");
  }
  if (opts.tiles.size() > 32) {
    die("too many tiles");
  }

  return opts;
}

void copy_blob(rt::IRuntime &runtime, rt::StreamId stream, std::byte *dst_base,
               uint64_t offset, const std::vector<std::byte> &blob,
               const std::string &name)
{
  if (offset + blob.size() > kErbiumMemSize) {
    die(name + " overflows Erbium memory buffer");
  }

  const auto evt = runtime.memcpyHostToDevice(stream, blob.data(),
                                              dst_base + offset, blob.size());
  if (!runtime.waitForEvent(evt)) {
    die("memcpy timed out for " + name);
  }
}

void dump_compact(rt::IRuntime &runtime, rt::StreamId stream,
                  std::byte *device_addr, const std::string &path)
{
  std::vector<std::byte> host(kCompactDumpSize);
  const auto evt = runtime.memcpyDeviceToHost(stream, device_addr, host.data(),
                                              host.size());

  if (!runtime.waitForEvent(evt)) {
    die("device-to-host memcpy timed out for " + path);
  }
  write_file(path, host);
}

} // namespace

int main(int argc, char **argv)
{
  const auto opts = parse_args(argc, argv);

  auto device_layer = dev::IDeviceLayer::createPcieDeviceLayer();
#ifdef WHISPER_RUNTIME_CREATE_RAW
  auto runtime = rt::IRuntime::create(device_layer.get());
#else
  auto runtime = rt::IRuntime::create(std::move(device_layer));
#endif

  runtime->setOnStreamErrorsCallback(
      [](rt::EventId id, const rt::StreamError &err) {
        std::cerr << "Stream error (event " << static_cast<int>(id)
                  << "): code " << static_cast<int>(err.errorCode_) << "\n";
      });
  runtime->setOnKernelAbortedErrorCallback(
      [](rt::EventId id, std::byte *, size_t, std::function<void()> free_res) {
        std::cerr << "Kernel aborted (event " << static_cast<int>(id)
                  << ")\n";
        free_res();
      });

  const auto devices = runtime->getDevices();
  if (devices.empty()) {
    die("no devices found");
  }
  const auto device = devices[0];

  const auto zero = read_file(opts.zero);
  const auto act = read_file(opts.act);
  const auto ln_weight = read_file(opts.ln_weight);
  const auto ln_bias = read_file(opts.ln_bias);
  const auto ln_ref = read_file(opts.ln_ref);

  std::vector<TileRuntime> tiles;
  tiles.reserve(opts.tiles.size());

  for (const auto &spec : opts.tiles) {
    TileRuntime tile;
    tile.spec = spec;
    tile.stream = runtime->createStream(device);
    tile.device_buf = runtime->mallocDevice(device, kErbiumMemSize);
    tile.device_buf_arg = reinterpret_cast<uintptr_t>(tile.device_buf);

    const auto elf = read_file(spec.elf);
    tile.load = runtime->loadCode(tile.stream, elf.data(), elf.size());
    if (!runtime->waitForEvent(tile.load.event_)) {
      die("ELF load timed out for " + spec.elf);
    }

    copy_blob(*runtime, tile.stream, tile.device_buf, kZeroOffset, zero,
              "zero");
    copy_blob(*runtime, tile.stream, tile.device_buf, kLnInOffset, act,
              "act");
    copy_blob(*runtime, tile.stream, tile.device_buf, kLnWeightOffset,
              ln_weight, "ln_weight");
    copy_blob(*runtime, tile.stream, tile.device_buf, kLnBiasOffset, ln_bias,
              "ln_bias");
    copy_blob(*runtime, tile.stream, tile.device_buf, kLnRefOffset, ln_ref,
              "ln_ref");

    const auto weight = read_file(spec.weight);
    copy_blob(*runtime, tile.stream, tile.device_buf, kWeightOffset, weight,
              spec.weight);
    tiles.push_back(std::move(tile));
  }

  rt::KernelLaunchOptions launch_opts;
  launch_opts.setBarrier(true);
  launch_opts.setFlushL3(true);

  const auto launch_start = std::chrono::steady_clock::now();
  for (auto &tile : tiles) {
    launch_opts.setShireMask(uint64_t{1} << tile.spec.shire);
    runtime->kernelLaunch(tile.stream, tile.load.kernel_,
                          reinterpret_cast<const std::byte *>(
                              &tile.device_buf_arg),
                          sizeof(tile.device_buf_arg), launch_opts);
  }

  const auto timeout = opts.timeout_secs == 0
                           ? std::chrono::hours(24)
                           : std::chrono::seconds(opts.timeout_secs);

  bool ok = true;
  for (auto &tile : tiles) {
    const bool wait_ok = runtime->waitForStream(tile.stream, timeout);
    const auto done = std::chrono::steady_clock::now();
    tile.wait_s =
        std::chrono::duration<double>(done - launch_start).count();

    if (!wait_ok) {
      tile.ok = false;
      tile.messages.push_back("Error: kernel execution timed out");
      runtime->abortStream(tile.stream);
    }

    const auto errors = runtime->retrieveStreamErrors(tile.stream);
    if (!errors.empty()) {
      tile.ok = false;
      tile.messages.push_back("Kernel finished with stream errors");
      for (const auto &err : errors) {
        tile.messages.push_back("Stream error code: " +
                                std::to_string(static_cast<int>(
                                    err.errorCode_)));
      }
    }

    if (tile.ok) {
      dump_compact(*runtime, tile.stream, tile.device_buf, tile.spec.dump);
    }
    write_log(tile);
    ok = ok && tile.ok;
  }

  for (auto &tile : tiles) {
    runtime->unloadCode(tile.load.kernel_);
    runtime->freeDevice(device, tile.device_buf);
    runtime->destroyStream(tile.stream);
  }

  const auto all_done = std::chrono::steady_clock::now();
  std::cout << "Parallel tail launcher wall seconds: "
            << std::chrono::duration<double>(all_done - launch_start).count()
            << "\n";

  return ok ? 0 : 1;
}
