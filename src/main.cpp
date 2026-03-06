// src/main.cpp
//
// Acoustic pedalboard (stereo audio output) with presets and distortion, reverb, and delay controls.
// Uses PortAudio for real-time audio input and output.
//
// Build:
//   cmake --build build -j
//
// Run:
//   ./build/acoustic_pedal
//   ./build/acoustic_pedal --in 2 --out 1
//
// Controls (type key, then press Enter):
//   q: quit
//   1: Studio Acoustic preset
//   2: Ambient Dreamy preset
//   3: Lo-Fi preset
//   4: Shoegaze preset
//
//   y: toggle delay on/off: adds echo
//   t: toggle distortion on/off: adds saturation and grit
//   r: toggle reverb on/off: adds atmospheric space
//
//   a/z: increase/decrease gain: overall volume
//
//   s/x: increase/decrease delay time: spacing between echoes
//   d/c: increase/decrease delay mix: loudness of echoes
//   f/v: increase/decrease delay feedback: number of repeating echoes
//
//   g/h: increase/decrease distortion drive: strength of saturation/fuzz
//   j/k: increase/decrease distortion tone: brightness of distorted tone
//
//   u/i: increase/decrease reverb mix: amount of ambient room sound
//   o/p: increase/decrease reverb decay: length of the reverb tail
//   l/;: increase/decrease reverb damping: brightness of the reverb tail
//
// Notes:
// - Use headphones (otherwise mic->speaker feedback can howl).
// - If you hear nothing, choose the correct device IDs with --in/--out.
// - Modular DSP blocks:
//     - Each block is a small self-contained processor with state + prepare() + process().
//     - The Pedalboard processes blocks in order: HPF -> Distortion -> Delay -> Reverb.

#include <portaudio.h>
#include <fstream>
#include <mutex>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

// Math helpers
static float clampf(float x, float lo, float hi) { return std::max(lo, std::min(hi, x)); }
static bool isFinite(float x) { return std::isfinite(x); }

// Parameters shared between main thread and audio callback
// Callback reads these values often, main thread updates them when user presses keys
// Atomics avoid data races without locks
struct Parameters {
  // Master gain
  std::atomic<float> gain{1.0f}; // 0..4

  // Delay
  std::atomic<bool>  delay_on{true};
  std::atomic<float> delay_ms{180.0f};       // 1..1200
  std::atomic<float> delay_mix{0.25f};       // 0..1
  std::atomic<float> delay_feedback{0.30f};  // 0..0.95

  // Distortion
  std::atomic<bool>  dist_on{false};
  std::atomic<float> dist_drive{2.5f}; // 1..10
  std::atomic<float> dist_tone{0.55f}; // 0..1 (0 dark, 1 bright)

  // Reverb
  std::atomic<bool>  rev_on{false};
  std::atomic<float> rev_mix{0.25f};   // 0..1
  std::atomic<float> rev_decay{0.55f}; // 0..0.95 (tail length)
  std::atomic<float> rev_damp{0.35f};  // 0..1 (0 bright, 1 dark)
};

// Copy current parameter values once per audio buffer
// Cheaper than loading atomics for every sample
struct ParamSnapshot {
  float gain;

  bool  delay_on;
  float delay_ms;
  float delay_mix;
  float delay_feedback;

  bool  dist_on;
  float dist_drive;
  float dist_tone;

  bool  rev_on;
  float rev_mix;
  float rev_decay;
  float rev_damp;
};

static ParamSnapshot snapshotParams(const Parameters& p) {
  ParamSnapshot s{};
  s.gain = p.gain.load();

  s.delay_on = p.delay_on.load();
  s.delay_ms = p.delay_ms.load();
  s.delay_mix = p.delay_mix.load();
  s.delay_feedback = p.delay_feedback.load();

  s.dist_on = p.dist_on.load();
  s.dist_drive = p.dist_drive.load();
  s.dist_tone = p.dist_tone.load();

  s.rev_on = p.rev_on.load();
  s.rev_mix = p.rev_mix.load();
  s.rev_decay = p.rev_decay.load();
  s.rev_damp = p.rev_damp.load();
  return s;
}

// DSP Block Interface:

// Each processing block:
// - prepares itself for a sample rate
// - processes one sample at a time
// - keeps its own internal state
struct IDspBlock {
  virtual ~IDspBlock() = default;
  virtual void prepare(double sampleRate) = 0;
  virtual float processSample(float x, const ParamSnapshot& ps) = 0;
};

// One-pole high-pass filter to remove low rumble/mud
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
    x1 = x;
    y1 = y;
    return y;
  }

  void reset() { x1 = 0.0f; y1 = 0.0f; }
};

// One-pole low-pass filter for tone shaping and reverb damping
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

  void reset() { y1 = 0.0f; }
};

// Block: input cleanup high-pass filter
// Rmoves low rumble and mud from input signal
struct HpfBlock final : public IDspBlock {
  double fs = 48000.0;
  OnePoleHPF hpf;

  float cutoffHz = 90.0f;

  void prepare(double sampleRate) override {
    fs = sampleRate;
    hpf.setCutoff(cutoffHz, (float)fs);
    hpf.reset();
  }

  float processSample(float x, const ParamSnapshot& /*ps*/) override {
    return hpf.process(x);
  }
};

// Block: soft-clip distortion with tone control
// A low-pass filter before distortion helps soften harsh guitar strum sound
// A second low-pass filter shapes brightness of the distorted sound
struct DistortionBlock final : public IDspBlock {
  double fs = 48000.0;
  OnePoleLPF toneLPF;
  OnePoleLPF preLPF;

  void prepare(double sampleRate) override {
    fs = sampleRate;
    toneLPF.setCutoff(6000.0f, (float)fs);
    toneLPF.reset();
    preLPF.setCutoff(2200.0f, (float)fs);
    preLPF.reset();
  }

  float processSample(float x, const ParamSnapshot& ps) override {
    if (!ps.dist_on) return x;

    // Clamp parameters to safe ranges
    float drive = clampf(ps.dist_drive, 1.0f, 10.0f);
    float tone01 = clampf(ps.dist_tone, 0.0f, 1.0f);

    // Waveshaping
    float softened = preLPF.process(x);
    float y = std::tanh(softened * drive);

    // Tone: adjust LPF cutoff for tone shaping
    float fc = 1200.0f + tone01 * 9000.0f; // 1.2k .. 10.2k
    toneLPF.setCutoff(fc, (float)fs);

    // Smooth the distorted signal
    y = toneLPF.process(y);
    return y;
  }
};

// Block: stereo delay, echo
// Uses separate left and right delay lines
// Delayed signal is mixed back with the dry signal
struct DelayLine {
  std::vector<float> buf;
  size_t write_idx = 0;

  void init(size_t size) {
    buf.assign(std::max<size_t>(1, size), 0.0f);
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

struct DelayBlock {
  double fs = 48000.0;
  DelayLine dlL;
  DelayLine dlR;
  size_t maxDelaySamples = 0;

  void prepare(double sampleRate) {
    fs = sampleRate;
    maxDelaySamples = (size_t)std::max(1.0, std::round(fs * 2.0));
    dlL.init(maxDelaySamples);
    dlR.init(maxDelaySamples);
  }

  // Process one mono input sample and return stereo output by reference
  void processStereo(float x, const ParamSnapshot& ps, float& outL, float& outR) {
    if (!ps.delay_on) {
      outL = x;
      outR = x;
      return;
    }

    float delay_ms = clampf(ps.delay_ms, 1.0f, 1200.0f);
    float mix = clampf(ps.delay_mix, 0.0f, 1.0f);
    float fb  = clampf(ps.delay_feedback, 0.0f, 0.95f);

    float delayMsL = delay_ms;
    float delayMsR = delay_ms;

    size_t delaySamplesL = (size_t)((delayMsL * 0.001) * fs);
    size_t delaySamplesR = (size_t)((delayMsR * 0.001) * fs);

    delaySamplesL = std::min(delaySamplesL, maxDelaySamples - 1);
    delaySamplesR = std::min(delaySamplesR, maxDelaySamples - 1);

    float delayedL = dlL.read(delaySamplesL);
    float delayedR = dlR.read(delaySamplesR);

    // Small crossfeed helps the stereo image feel more connected
    float writeL = x + (0.85f * delayedL + 0.15f * delayedR) * fb;
    float writeR = x + (0.85f * delayedR + 0.15f * delayedL) * fb;

    writeL = clampf(writeL, -1.0f, 1.0f);
    writeR = clampf(writeR, -1.0f, 1.0f);

    dlL.write(writeL);
    dlR.write(writeR);

    outL = x * (1.0f - mix) + delayedL * mix;
    outR = x * (1.0f - mix) + delayedR * mix;
  }
};

// Block: stereo Schroeder-style reverb
// Uses comb filters for dense echoes and allpass filters for diffusion
// Separate left and right tanks create stereo width
struct Comb {
  std::vector<float> buf;
  size_t idx = 0;
  OnePoleLPF dampLPF;
  double fs = 48000.0;

  void init(size_t delaySamples, double sampleRate) {
    fs = sampleRate;
    buf.assign(std::max<size_t>(1, delaySamples), 0.0f);
    idx = 0;
    dampLPF.setCutoff(6000.0f, (float)fs);
    dampLPF.reset();
  }

  float process(float x, float feedback, float damp01) {
    float y = buf[idx];

    // Damping (darkens the tail as damp increases)
    float fc = 1200.0f + (1.0f - damp01) * 9000.0f; // damp=1 -> ~1.2k, damp=0 -> ~10k
    dampLPF.setCutoff(fc, (float)fs);

    float fbSignal = dampLPF.process(y);
    buf[idx] = x + fbSignal * feedback;

    idx = (idx + 1) % buf.size();
    return y;
  }
};

struct ModulatedAllpass {
  std::vector<float> buf;
  size_t write_idx = 0;

  float g = 0.5f;

  // Base delay in samples
  float baseDelay = 100.0f;

  // Modulation settings
  float modDepth = 4.0f;   // In samples
  float modRate = 0.15f;   // Hz
  float phase = 0.0f;
  float fs = 48000.0f;

  void init(size_t delaySamples, float gain, float sampleRate,
            float depthSamples = 4.0f, float rateHz = 0.15f, float startPhase = 0.0f) {
    buf.assign(std::max<size_t>(1, delaySamples + 32), 0.0f);
    write_idx = 0;
    g = gain;
    baseDelay = (float)delaySamples;
    fs = sampleRate;
    modDepth = depthSamples;
    modRate = rateHz;
    phase = startPhase;
  }

  float process(float x) {
    // Slow sinusoidal modulation
    float mod = std::sin(phase) * modDepth;
    phase += 2.0f * 3.14159265f * modRate / fs;
    if (phase > 2.0f * 3.14159265f) phase -= 2.0f * 3.14159265f;

    float readDelay = baseDelay + mod;
    if (readDelay < 1.0f) readDelay = 1.0f;

    // Fractional delay read with linear interpolation
    float readPos = (float)write_idx - readDelay;
    while (readPos < 0.0f) readPos += (float)buf.size();

    int i0 = (int)readPos;
    int i1 = (i0 + 1) % (int)buf.size();
    float frac = readPos - (float)i0;

    float y = buf[i0] * (1.0f - frac) + buf[i1] * frac;

    // Classic allpass formula
    float out = -g * x + y;
    buf[write_idx] = x + g * out;

    write_idx = (write_idx + 1) % buf.size();
    return out;
  }
};struct ReverbBlock {
  double fs = 48000.0;

  // Separate left/right tanks
  Comb combsL[4];
  Comb combsR[4];
  ModulatedAllpass ap1L, ap2L;
  ModulatedAllpass ap1R, ap2R;

  void prepare(double sampleRate) {
    fs = sampleRate;

    auto scale = [&](int base48k) -> size_t {
      return (size_t)std::max(1.0, std::round((double)base48k * (fs / 48000.0)));
    };

    // Slightly different comb lengths L/R for width
    combsL[0].init(scale(1557), fs);
    combsL[1].init(scale(1617), fs);
    combsL[2].init(scale(1491), fs);
    combsL[3].init(scale(1422), fs);

    combsR[0].init(scale(1589), fs);
    combsR[1].init(scale(1649), fs);
    combsR[2].init(scale(1513), fs);
    combsR[3].init(scale(1453), fs);

    // Modulated allpasses with slightly different settings L/R
    ap1L.init(scale(225), 0.68f, (float)fs, 5.0f, 0.11f, 0.0f);
    ap2L.init(scale(556), 0.68f, (float)fs, 7.0f, 0.07f, 1.7f);

    ap1R.init(scale(248), 0.68f, (float)fs, 6.0f, 0.09f, 0.9f);
    ap2R.init(scale(579), 0.68f, (float)fs, 8.0f, 0.06f, 2.4f);
  }

  void processStereo(float inL, float inR, const ParamSnapshot& ps, float& outL, float& outR) {
    if (!ps.rev_on) {
      outL = inL;
      outR = inR;
      return;
    }

    float mix = clampf(ps.rev_mix, 0.0f, 1.0f);
    float decay = clampf(ps.rev_decay, 0.0f, 0.95f);
    float damp = clampf(ps.rev_damp, 0.0f, 1.0f);

    // Soften transients into the tank
    float tankInL = std::tanh(inL * 0.18f * 1.5f);
    float tankInR = std::tanh(inR * 0.18f * 1.5f);

    // Parallel combs
    float sumL = 0.0f;
    float sumR = 0.0f;

    sumL += combsL[0].process(tankInL, decay, damp);
    sumL += combsL[1].process(tankInL, decay, damp);
    sumL += combsL[2].process(tankInL, decay, damp);
    sumL += combsL[3].process(tankInL, decay, damp);

    sumR += combsR[0].process(tankInR, decay, damp);
    sumR += combsR[1].process(tankInR, decay, damp);
    sumR += combsR[2].process(tankInR, decay, damp);
    sumR += combsR[3].process(tankInR, decay, damp);

    sumL *= 0.20f;
    sumR *= 0.20f;

    // Small crossfeed between tanks makes it wider and smoother
    float crossL = 0.85f * sumL + 0.15f * sumR;
    float crossR = 0.85f * sumR + 0.15f * sumL;

    // Diffusion
    float wetL = ap1L.process(crossL);
    wetL = ap2L.process(wetL);

    float wetR = ap1R.process(crossR);
    wetR = ap2R.process(wetR);

    outL = inL * (1.0f - mix) + wetL * mix;
    outR = inR * (1.0f - mix) + wetR * mix;
  }
};

// Pedalboard: chains blocks in order
struct Pedalboard {
  // Store pointers to blocks that live elsewhere (AudioState)
  // No allocations in audio callback, set this up once
  std::vector<IDspBlock*> chain;

  void prepare(double fs) {
    for (auto* b : chain) b->prepare(fs);
  }

  float process(float x, const ParamSnapshot& ps) {
    for (auto* b : chain) x = b->processSample(x, ps);
    return x;
  }
};

// Stores processed stereo output, optionally saves it as a .wav file
struct Recorder {
  std::vector<float> samples;
  std::mutex mtx;
  bool enabled = false;

  void pushStereo(float left, float right) {
    if (!enabled) return;
    std::lock_guard<std::mutex> lock(mtx);
    samples.push_back(left);
    samples.push_back(right);
  }

  static void writeWav16(const std::string& filename,
                         const std::vector<float>& interleavedStereo,
                         int sampleRate) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
      std::cerr << "Could not open WAV file for writing: " << filename << "\n";
      return;
    }

    int numChannels = 2;
    int bitsPerSample = 16;
    int byteRate = sampleRate * numChannels * bitsPerSample / 8;
    int blockAlign = numChannels * bitsPerSample / 8;
    int dataSize = static_cast<int>(interleavedStereo.size()) * sizeof(int16_t);
    int chunkSize = 36 + dataSize;

    // RIFF header
    out.write("RIFF", 4);
    out.write(reinterpret_cast<const char*>(&chunkSize), 4);
    out.write("WAVE", 4);

    // fmt chunk
    out.write("fmt ", 4);
    int subchunk1Size = 16;
    short audioFormat = 1; // PCM
    short channels = static_cast<short>(numChannels);
    short bps = static_cast<short>(bitsPerSample);
    short align = static_cast<short>(blockAlign);

    out.write(reinterpret_cast<const char*>(&subchunk1Size), 4);
    out.write(reinterpret_cast<const char*>(&audioFormat), 2);
    out.write(reinterpret_cast<const char*>(&channels), 2);
    out.write(reinterpret_cast<const char*>(&sampleRate), 4);
    out.write(reinterpret_cast<const char*>(&byteRate), 4);
    out.write(reinterpret_cast<const char*>(&align), 2);
    out.write(reinterpret_cast<const char*>(&bps), 2);

    // data chunk
    out.write("data", 4);
    out.write(reinterpret_cast<const char*>(&dataSize), 4);

    for (float x : interleavedStereo) {
      float clamped = clampf(x, -1.0f, 1.0f);
      int16_t s = static_cast<int16_t>(clamped * 32767.0f);
      out.write(reinterpret_cast<const char*>(&s), sizeof(int16_t));
    }

    out.close();
  }

  void saveToFile(const std::string& filename, int sampleRate) {
    std::lock_guard<std::mutex> lock(mtx);
    writeWav16(filename, samples, sampleRate);
  }
};

// AudioState: owns all blocks + pedalboard + parameters
struct AudioState {
  double sample_rate = 48000.0;

  Parameters params;

  // Blocks
  HpfBlock hpf;
  DistortionBlock dist;
  DelayBlock delay;
  ReverbBlock reverb;

  // Pedalboard chain
  Pedalboard board;

  Recorder recorder;
};

// PortAudio error helper
static void paCheck(PaError err, const char* msg) {
  if (err != paNoError) {
    std::cerr << "PortAudio error: " << msg << " : " << Pa_GetErrorText(err) << "\n";
    std::exit(1);
  }
}

// Presets
enum class Preset {
  StudioAcoustic = 1,
  AmbientDreamy = 2,
  LoFi = 3,
  Shoegaze = 4
};

// Apply 4 different preset settings
// Gain values are intentionally adjusted so presets sound closer in perceived volume
static void applyPreset(AudioState& st, Preset p) {
  auto& par = st.params;

  switch (p) {
    case Preset::StudioAcoustic:
      par.gain.store(2.7f);

      par.dist_on.store(false);
      par.dist_drive.store(0.0f);
      par.dist_tone.store(0.0f);

      par.delay_on.store(true);
      par.delay_ms.store(140.0f);
      par.delay_mix.store(0.12f);
      par.delay_feedback.store(0.18f);

      par.rev_on.store(true);
      par.rev_mix.store(0.18f);
      par.rev_decay.store(0.40f);
      par.rev_damp.store(0.45f);
      break;

    case Preset::AmbientDreamy:
      par.gain.store(4.0f);

      par.dist_on.store(false);
      par.dist_drive.store(0.0f);
      par.dist_tone.store(0.0f);

      par.delay_on.store(true);
      par.delay_ms.store(200.0f);
      par.delay_mix.store(0.10f);
      par.delay_feedback.store(0.35f);

      par.rev_on.store(true);
      par.rev_mix.store(0.90f);
      par.rev_decay.store(0.95f);
      par.rev_damp.store(0.88f);
      break;

    case Preset::LoFi:
      par.gain.store(1.6f);

      par.dist_on.store(true);
      par.dist_drive.store(3.0f);
      par.dist_tone.store(0.05f);

      par.delay_on.store(true);
      par.delay_ms.store(110.0f);
      par.delay_mix.store(0.18f);
      par.delay_feedback.store(0.22f);

      par.rev_on.store(false);
      par.rev_mix.store(0.0f);
      par.rev_decay.store(0.0f);
      par.rev_damp.store(0.0f);
      break;

    case Preset::Shoegaze:
      par.gain.store(2.7f);

      par.dist_on.store(true);
      par.dist_drive.store(8.0f);
      par.dist_tone.store(0.5f);

      par.delay_on.store(true);
      par.delay_ms.store(70.0f);
      par.delay_mix.store(0.10f);
      par.delay_feedback.store(0.28f);

      par.rev_on.store(true);
      par.rev_mix.store(1.0f);
      par.rev_decay.store(0.96f);
      par.rev_damp.store(0.90f);
      break;
  }
}

// Device utilities
static void printDevices() {
  int n = Pa_GetDeviceCount();
  if (n < 0) paCheck(n, "Pa_GetDeviceCount");

  std::cout << "Devices:\n";
  for (int i = 0; i < n; ++i) {
    const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
    std::cout << "  [" << i << "] " << info->name
              << " (in=" << info->maxInputChannels
              << ", out=" << info->maxOutputChannels << ")\n";
  }
}

static void parseDeviceFlags(int argc, char** argv, int& inDev, int& outDev) {
  for (int i = 1; i + 1 < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--in")  inDev  = std::stoi(argv[i + 1]);
    if (arg == "--out") outDev = std::stoi(argv[i + 1]);
  }
}

// Audio callback (real-time thread)
static int audioCallback(const void* inputBuffer, void* outputBuffer,
                         unsigned long framesPerBuffer,
                         const PaStreamCallbackTimeInfo* /*timeInfo*/,
                         PaStreamCallbackFlags /*statusFlags*/,
                         void* userData) {
  auto* st = static_cast<AudioState*>(userData);
  const float* in = static_cast<const float*>(inputBuffer);
  float* out = static_cast<float*>(outputBuffer);

  bool hasInput = (in != nullptr);

  ParamSnapshot ps = snapshotParams(st->params);
  ps.gain = clampf(ps.gain, 0.0f, 4.0f);

  for (unsigned long i = 0; i < framesPerBuffer; ++i) {
    float x = hasInput ? in[i] : 0.0f;

    // Master gain
    x *= ps.gain;

    // Mono pre-chain: HPF -> Distortion
    float mono = st->board.process(x, ps);

    // Stereo delay
    float afterDelayL = mono;
    float afterDelayR = mono;
    st->delay.processStereo(mono, ps, afterDelayL, afterDelayR);

    // Stereo reverb
    float finalL = afterDelayL;
    float finalR = afterDelayR;
    st->reverb.processStereo(afterDelayL, afterDelayR, ps, finalL, finalR);

    finalL = clampf(finalL, -1.0f, 1.0f);
    finalR = clampf(finalR, -1.0f, 1.0f);

    if (!isFinite(finalL)) finalL = 0.0f;
    if (!isFinite(finalR)) finalR = 0.0f;

    // Save processed stereo output for WAV recording
    st->recorder.pushStereo(finalL, finalR);

    out[2 * i]     = finalL;
    out[2 * i + 1] = finalR;
  }

  return paContinue;
}

std::string makeBar(float value, float min, float max, int width = 14) {
  value = clampf(value, min, max);
  float norm = (value - min) / (max - min);

  int filled = (int)(norm * width);

  std::string bar = "[";
  for (int i = 0; i < width; i++) {
    if (i < filled)
      bar += "█";
    else
      bar += "-";
  }
  bar += "]";

  return bar;
}

void drawUI(const ParamSnapshot& ps) {

  std::cout << "\033[2J\033[H";

  std::cout << "========================================\n";
  std::cout << "        Acoustic Pedalboard\n";
  std::cout << "========================================\n\n";

  std::cout << "GAIN\n";
  std::cout << "  " << makeBar(ps.gain, 0.0f, 4.0f)
            << "  " << ps.gain << "\n\n";

  std::cout << "DELAY  " << (ps.delay_on ? "ON" : "OFF") << "\n";

  std::cout << "  time      "
            << makeBar(ps.delay_ms, 1.0f, 1200.0f)
            << "  " << ps.delay_ms << " ms\n";

  std::cout << "  mix       "
            << makeBar(ps.delay_mix, 0.0f, 1.0f)
            << "  " << ps.delay_mix << "\n";

  std::cout << "  feedback  "
            << makeBar(ps.delay_feedback, 0.0f, 0.95f)
            << "  " << ps.delay_feedback << "\n\n";

  std::cout << "DISTORTION  " << (ps.dist_on ? "ON" : "OFF") << "\n";

  std::cout << "  drive     "
            << makeBar(ps.dist_drive, 1.0f, 10.0f)
            << "  " << ps.dist_drive << "\n";

  std::cout << "  tone      "
            << makeBar(ps.dist_tone, 0.0f, 1.0f)
            << "  " << ps.dist_tone << "\n\n";

  std::cout << "REVERB  " << (ps.rev_on ? "ON" : "OFF") << "\n";

  std::cout << "  mix       "
            << makeBar(ps.rev_mix, 0.0f, 1.0f)
            << "  " << ps.rev_mix << "\n";

  std::cout << "  decay     "
            << makeBar(ps.rev_decay, 0.0f, 0.95f)
            << "  " << ps.rev_decay << "\n";

  std::cout << "  damp      "
            << makeBar(ps.rev_damp, 0.0f, 1.0f)
            << "  " << ps.rev_damp << "\n\n";

  std::cout << "Press key then Enter (q to quit)\n";
}

int main(int argc, char** argv) {
  paCheck(Pa_Initialize(), "Pa_Initialize");

  printDevices();

  int inDev = Pa_GetDefaultInputDevice();
  int outDev = Pa_GetDefaultOutputDevice();
  if (inDev == paNoDevice || outDev == paNoDevice) {
    std::cerr << "No default input/output device.\n";
    Pa_Terminate();
    return 1;
  }

  parseDeviceFlags(argc, argv, inDev, outDev);

  PaStreamParameters inputParams{};
  PaStreamParameters outputParams{};

  inputParams.device = inDev;
  outputParams.device = outDev;

  const PaDeviceInfo* inInfo = Pa_GetDeviceInfo(inDev);
  const PaDeviceInfo* outInfo = Pa_GetDeviceInfo(outDev);

  inputParams.channelCount = 1;
  inputParams.sampleFormat = paFloat32;
  inputParams.suggestedLatency = inInfo->defaultLowInputLatency;
  inputParams.hostApiSpecificStreamInfo = nullptr;

  outputParams.channelCount = 2;
  outputParams.sampleFormat = paFloat32;
  outputParams.suggestedLatency = outInfo->defaultLowOutputLatency;
  outputParams.hostApiSpecificStreamInfo = nullptr;

  AudioState st;

  // Ask user if they want to record
  char recordChoice;
  std::cout << "Record session to recording.wav? (y/n): ";
  std::cin >> recordChoice;

  if (recordChoice == 'y' || recordChoice == 'Y') {
    st.recorder.enabled = true;
    std::cout << "Recording enabled.\n";
  } else {
    st.recorder.enabled = false;
    std::cout << "Recording disabled.\n";
  }

  // Set up the pedalboard chain (order matters)
  st.board.chain = {
    &st.hpf,
    &st.dist
  };
  
  // Apply default preset StudioAcoustic
  applyPreset(st, Preset::StudioAcoustic);

  // Try 48k first, fallback to device default if needed.
  st.sample_rate = 48000.0;
  unsigned long framesPerBuffer = 256;

  // Prepare blocks (allocates internal buffers for delay/reverb, etc.)
  st.board.prepare(st.sample_rate);
  st.delay.prepare(st.sample_rate);
  st.reverb.prepare(st.sample_rate);

  PaStream* stream = nullptr;
  PaError openErr = Pa_OpenStream(
      &stream,
      &inputParams,
      &outputParams,
      st.sample_rate,
      framesPerBuffer,
      paNoFlag,
      audioCallback,
      &st);

  if (openErr != paNoError) {
    std::cerr << "OpenStream failed at 48000 Hz, trying device default...\n";
    st.sample_rate = outInfo->defaultSampleRate;

    // Re-prepare DSP blocks for new sample rate
    st.board.prepare(st.sample_rate);
    st.delay.prepare(st.sample_rate);
    st.reverb.prepare(st.sample_rate);

    paCheck(Pa_OpenStream(
        &stream,
        &inputParams,
        &outputParams,
        st.sample_rate,
        framesPerBuffer,
        paNoFlag,
        audioCallback,
        &st), "Pa_OpenStream fallback");
  }

  paCheck(Pa_StartStream(stream), "Pa_StartStream");

std::cout
  << "\nRunning acoustic_pedal (stereo output)\n"
  << "Input:  [" << inDev << "] " << inInfo->name << "\n"
  << "Output: [" << outDev << "] " << outInfo->name << "\n"
  << "Sample rate: " << st.sample_rate << " Hz\n"
  << "Frames/buffer: " << framesPerBuffer << "\n\n"

  << "Presets:\n"
  << "  1 Studio Acoustic : clean guitar with light ambience\n"
  << "  2 Ambient Dreamy  : distant, soft sound with reverb wash\n"
  << "  3 Lo-Fi           : darker, saturated tone with grit\n"
  << "  4 Shoegaze        : thick, distorted wall of noise\n\n"

  << "Toggles:\n"
  << "  y delay on/off      : adds echo\n"
  << "  t distortion on/off : adds saturation and grit\n"
  << "  r reverb on/off     : adds atmospheric space\n\n"

  << "Knobs:\n"
  << "Gain:\n"
  << "  a/z  gain +/-           : overall volume\n"
  << "Delay:\n"
  << "  s/x  delay_ms +/-       : spacing between echoes\n"
  << "  d/c  delay_mix +/-      : loudness of echoes\n"
  << "  f/v  delay_feedback +/- : number of repeating echoes\n"
  << "Distortion:\n"
  << "  g/h  dist_drive +/-     : strength of saturation/fuzz\n"
  << "  j/k  dist_tone +/-      : brightness of distorted tone\n"
  << "Reverb:\n"
  << "  u/i  rev_mix +/-        : amount of ambient room sound\n"
  << "  o/p  rev_decay +/-      : length of the reverb tail\n"
  << "  l/;  rev_damp +/-       : brightness of the reverb tail\n\n"

  << "q = quit\n\n"

  << "Use headphones to avoid microphone feedback.\n\n";

  // Redraw UI
  drawUI(snapshotParams(st.params));
  
  // Main thread control loop:
  // Read a single char each time, then update atomics
  // Requires Enter after each char bc uses std::cin >> ch
  std::cin.clear();
  bool running = true;
  while (running) {
    char ch;
    std::cin >> ch;

    // Load current values into local variables, modify, then store back
    float gain = st.params.gain.load();

    bool delay_on = st.params.delay_on.load();
    float dms = st.params.delay_ms.load();
    float dmx = st.params.delay_mix.load();
    float dfb = st.params.delay_feedback.load();

    bool dist_on = st.params.dist_on.load();
    float drive = st.params.dist_drive.load();
    float tone = st.params.dist_tone.load();

    bool rev_on = st.params.rev_on.load();
    float rmx = st.params.rev_mix.load();
    float rdec = st.params.rev_decay.load();
    float rdamp = st.params.rev_damp.load();

    // Redraw UI after parameter changes
    drawUI(snapshotParams(st.params));

    bool appliedPreset = false;
    switch (ch) {
      case 'q': running = false; break;

      // Presets
      case '1': applyPreset(st, Preset::StudioAcoustic); appliedPreset = true; break;
      case '2': applyPreset(st, Preset::AmbientDreamy); appliedPreset = true; break;
      case '3': applyPreset(st, Preset::LoFi); appliedPreset = true; break;
      case '4': applyPreset(st, Preset::Shoegaze); appliedPreset = true; break;

      // Toggles
      case 'y': delay_on = !delay_on; break;
      case 't': dist_on = !dist_on; break;
      case 'r': rev_on = !rev_on; break;

      // Gain
      case 'a': gain += 0.05f; break;
      case 'z': gain -= 0.05f; break;

      // Delay knobs
      case 's': dms += 10.0f; break;
      case 'x': dms -= 10.0f; break;
      case 'd': dmx += 0.05f; break;
      case 'c': dmx -= 0.05f; break;
      case 'f': dfb += 0.05f; break;
      case 'v': dfb -= 0.05f; break;

      // Distortion knobs
      case 'g': drive += 0.25f; break;
      case 'h': drive -= 0.25f; break;
      case 'j': tone += 0.05f; break;
      case 'k': tone -= 0.05f; break;

      // Reverb knobs
      case 'u': rmx += 0.05f; break;
      case 'i': rmx -= 0.05f; break;
      case 'o': rdec += 0.05f; break;
      case 'p': rdec -= 0.05f; break;
      case 'l': rdamp += 0.05f; break;
      case ';': rdamp -= 0.05f; break;

      default: break;
    }

    // If applied preset, already updated atomics, redraw UI and continue
    if (appliedPreset) {
      drawUI(snapshotParams(st.params));
      continue;
    }
    
    // Clamp ranges (prevents runaway/weirdness)
    gain = clampf(gain, 0.0f, 4.0f);

    dms = clampf(dms, 1.0f, 1200.0f);
    dmx = clampf(dmx, 0.0f, 1.0f);
    dfb = clampf(dfb, 0.0f, 0.95f);

    drive = clampf(drive, 1.0f, 10.0f);
    tone  = clampf(tone, 0.0f, 1.0f);

    rmx   = clampf(rmx, 0.0f, 1.0f);
    rdec  = clampf(rdec, 0.0f, 0.95f);
    rdamp = clampf(rdamp, 0.0f, 1.0f);

    // Store updated values back to shared parameters
    st.params.gain.store(gain);

    st.params.delay_on.store(delay_on);
    st.params.delay_ms.store(dms);
    st.params.delay_mix.store(dmx);
    st.params.delay_feedback.store(dfb);

    st.params.dist_on.store(dist_on);
    st.params.dist_drive.store(drive);
    st.params.dist_tone.store(tone);

    st.params.rev_on.store(rev_on);
    st.params.rev_mix.store(rmx);
    st.params.rev_decay.store(rdec);
    st.params.rev_damp.store(rdamp);
  }
  
  if (st.recorder.enabled) {
    std::cout << "Saving recording to recording.wav ...\n";
    st.recorder.saveToFile("recording.wav", (int)st.sample_rate);
  }

  paCheck(Pa_StopStream(stream), "Pa_StopStream");
  paCheck(Pa_CloseStream(stream), "Pa_CloseStream");
  paCheck(Pa_Terminate(), "Pa_Terminate");

  std::cout << "Sounds good! See you later ^_^\n";
  return 0;
}