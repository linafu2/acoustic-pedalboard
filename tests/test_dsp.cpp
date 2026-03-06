// tests/test_dsp.cpp
//
// Basic unit tests for DSP blocks.
// These are “sanity tests” — not perfect audio-quality tests, but they catch:
// - explosions / instability
// - NaNs
// - obviously broken math
// - shape sanity for distortion curve
//
// Run:
//   cmake --build build -j
//   ./build/run_tests

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

// Small helpers
static bool isFinite(float x) { return std::isfinite(x); }
static float clampf(float x, float lo, float hi) { return std::max(lo, std::min(hi, x)); }

// ---- Minimal copies of core DSP pieces we want to test ----
// (In a bigger project you'd move these into a shared header/library.
// For now, keeping tests independent keeps it simple.)

struct OnePoleHPF {
  float a = 0.0f;
  float x1 = 0.0f;
  float y1 = 0.0f;

  void setCutoff(float fc, float fs) {
    float rc = 1.0f / (2.0f * 3.14159265f * fc);
    float dt = 1.0f / fs;
    a = rc / (rc + dt);
  }

  float process(float x) {
    float y = a * (y1 + x - x1);
    x1 = x; y1 = y;
    return y;
  }
};

struct OnePoleLPF {
  float a = 0.0f;
  float y1 = 0.0f;

  void setCutoff(float fc, float fs) {
    float rc = 1.0f / (2.0f * 3.14159265f * fc);
    float dt = 1.0f / fs;
    a = dt / (rc + dt);
  }

  float process(float x) {
    y1 = y1 + a * (x - y1);
    return y1;
  }
};

// Distortion curve under test: tanh(x * drive)
static float distort(float x, float drive) {
  drive = clampf(drive, 1.0f, 10.0f);
  return std::tanh(x * drive);
}

// DelayLine wrap sanity
struct DelayLine {
  std::vector<float> buf;
  size_t write_idx = 0;

  void init(size_t n) {
    buf.assign(std::max<size_t>(1, n), 0.0f);
    write_idx = 0;
  }

  float read(size_t delay_samples) const {
    size_t N = buf.size();
    size_t idx = (write_idx + N - (delay_samples % N)) % N;
    return buf[idx];
  }

  void write(float x) {
    buf[write_idx] = x;
    write_idx = (write_idx + 1) % buf.size();
  }
};

// Very small reverb stability proxy: a comb filter
struct Comb {
  std::vector<float> buf;
  size_t idx = 0;

  void init(size_t delaySamples) {
    buf.assign(std::max<size_t>(1, delaySamples), 0.0f);
    idx = 0;
  }

  float process(float x, float feedback) {
    float y = buf[idx];
    buf[idx] = x + y * feedback;
    idx = (idx + 1) % buf.size();
    return y;
  }
};

// ---- Tests ----

static void test_distortion_curve() {
  // 1) Output should always be within [-1,1]
  for (float drive : {1.0f, 2.0f, 5.0f, 10.0f}) {
    for (float x : {-10.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 10.0f}) {
      float y = distort(x, drive);
      assert(isFinite(y));
      assert(y >= -1.0f - 1e-6f && y <= 1.0f + 1e-6f);
    }
  }

  // 2) Odd symmetry-ish: tanh is odd, so f(-x) == -f(x)
  for (float drive : {1.0f, 3.0f, 8.0f}) {
    for (float x : {0.1f, 0.3f, 0.7f, 1.5f}) {
      float y1 = distort(x, drive);
      float y2 = distort(-x, drive);
      assert(std::fabs(y2 + y1) < 1e-5f);
    }
  }

  // 3) Monotonic for positive x (increasing input increases output)
  // (tanh is strictly increasing)
  float drive = 4.0f;
  float prev = distort(0.0f, drive);
  for (int i = 1; i <= 100; ++i) {
    float x = i / 20.0f; // 0.05 .. 5.0
    float y = distort(x, drive);
    assert(y >= prev - 1e-6f);
    prev = y;
  }
}

static void test_filter_stability() {
  // HPF/LPF should not produce NaNs or explode for a bounded input.
  const float fs = 48000.0f;

  OnePoleHPF hpf;
  hpf.setCutoff(90.0f, fs);

  OnePoleLPF lpf;
  lpf.setCutoff(3000.0f, fs);

  // Feed a step input and ensure outputs remain finite/bounded.
  float x = 1.0f;
  float maxAbsHPF = 0.0f;
  float maxAbsLPF = 0.0f;

  for (int n = 0; n < 200000; ++n) { // ~4 seconds at 48k
    float yh = hpf.process(x);
    float yl = lpf.process(x);

    assert(isFinite(yh));
    assert(isFinite(yl));

    maxAbsHPF = std::max(maxAbsHPF, std::fabs(yh));
    maxAbsLPF = std::max(maxAbsLPF, std::fabs(yl));

    // Neither should blow up to huge values for a constant bounded input.
    // These thresholds are generous.
    assert(maxAbsHPF < 10.0f);
    assert(maxAbsLPF < 10.0f);
  }

  // LPF on step should approach ~1.0 (not required exactly, but sanity)
  float tail = lpf.process(1.0f);
  assert(tail > 0.5f && tail <= 1.1f);
}

static void test_delay_wrap() {
  DelayLine dl;
  dl.init(8);

  // Write 0..7
  for (int i = 0; i < 8; ++i) dl.write((float)i);

  // Now write_idx has wrapped to 0.
  // Reading delay 1 should give last written = 7
  float y = dl.read(1);
  assert(std::fabs(y - 7.0f) < 1e-6f);

  // Write 8, write_idx=1, last written 8 at buf[0]
  dl.write(8.0f);
  float y1 = dl.read(1);
  assert(std::fabs(y1 - 8.0f) < 1e-6f);
}

static void test_reverb_stability_proxy() {
  // A simple comb with feedback < 1 should not explode.
  Comb c;
  c.init(1000);

  float fb = 0.8f;
  float maxAbs = 0.0f;

  // Impulse input: 1 then zeros
  for (int n = 0; n < 200000; ++n) {
    float x = (n == 0) ? 1.0f : 0.0f;
    float y = c.process(x, fb);

    assert(isFinite(y));
    maxAbs = std::max(maxAbs, std::fabs(y));

    // Should remain bounded.
    assert(maxAbs < 10.0f);
  }
}

int main() {
  std::cout << "Running DSP tests...\n";

  test_distortion_curve();
  std::cout << "  distortion curve: OK\n";

  test_filter_stability();
  std::cout << "  filter stability: OK\n";

  test_delay_wrap();
  std::cout << "  delay wrap: OK\n";

  test_reverb_stability_proxy();
  std::cout << "  reverb stability proxy: OK\n";

  std::cout << "All tests passed.\n";
  return 0;
}