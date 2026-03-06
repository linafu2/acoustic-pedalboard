# Acoustic Pedalboard

A real-time acoustic guitar effects pedalboard built in **C++** with **PortAudio**.  
This project captures live audio input, processes it through a modular DSP pipeline, and outputs **stereo audio** with adjustable **distortion, delay, and reverb**.

It also supports:
- **preset switching**
- **live parameter control**
- **optional WAV recording**

---

## Features

- **Real-time audio processing** with PortAudio
- **Stereo output**
- **Modular DSP architecture**
- **Input cleanup high-pass filter**
- **Soft-clip distortion with tone control**
- **Stereo delay**
- **Stereo modulated reverb**
- **Four presets**
  - Studio Acoustic
  - Ambient Dreamy
  - Lo-Fi
  - Shoegaze
- **Optional recording** to `recording.wav`
- **Live parameter control while audio is running**
- **Terminal UI with real-time parameter visualizer**

---

## Terminal UI

The program includes a **terminal-based user interface** that displays the current effect settings and visualizes parameter values using bar indicators.
The interface updates whenever parameters change, allowing quick visual feedback while adjusting effects.

---

## Relevancy

I created this project to easily record and apply audio effects while playing acoustic guitar with only a laptop, mainly for personal use.

I also explored audio software engineering concepts, including:
- low-latency audio I/O
- real-time callback design
- modular DSP block design
- stereo effect processing
- preset management
- safe parameter updates with atomics
- WAV file export

---

## Signal Chain

The audio processing pipeline is:

input microphone
→ master gain
→ high-pass filter
→ distortion
→ stereo delay
→ stereo reverb
→ stereo output

Each stage is implemented as an independent DSP block, making the processing chain modular and easy to extend.

---

## Architecture

Each effect is implemented as a self-contained DSP block containing:

- internal state
- `prepare(sampleRate)`
- `processSample()`

The pedalboard chains blocks together in the following order:
HPF -> Distortion -> Delay -> Reverb.

This modular design allows additional effects to be added easily.

---

## Build

cmake -B build
cmake --build build -j

---

## Run

./build/acoustic_pedal
Optional device selection:
./build/acoustic_pedal --in 2 --out 1

---

## Controls

While the program is running:

### Presets
1: Studio Acoustic
2: Ambient Dreamy
3: Lo-Fi
4: Shoegaze

### Effect Toggles
y: toggle delay on/off: adds echo
t: toggle distortion on/off: adds saturation and grit
r: toggle reverb on/off: adds atmospheric space

### Gain
a/z: increase/decrease gain: overall volume

### Delay
s/x: increase/decrease delay time: spacing between echoes
d/c: increase/decrease delay mix: loudness of echoes
f/v: increase/decrease delay feedback: number of repeating echoes

### Distortion
g/h: increase/decrease distortion drive: strength of saturation/fuzz
j/k: increase/decrease distortion tone: brightness of distorted tone

### Reverb
u/i: increase/decrease reverb mix: amount of ambient room sound
o/p: increase/decrease reverb decay: length of the reverb tail
l/;: increase/decrease reverb damping: brightness of the reverb tail

q: quit

---

## Recording

When starting the program, you can optionally enable recording.
If enabled, the processed stereo output will be saved to: recording.wav.