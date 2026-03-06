cmake_minimum_required(VERSION 3.16)
project(acoustic_pedal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Main app
add_executable(acoustic_pedal
  src/main.cpp
)

# PortAudio via pkg-config
find_package(PkgConfig REQUIRED)

if(APPLE AND EXISTS "/opt/homebrew/lib/pkgconfig")
  set(ENV{PKG_CONFIG_PATH} "/opt/homebrew/lib/pkgconfig:/opt/homebrew/share/pkgconfig:$ENV{PKG_CONFIG_PATH}")
endif()

pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)

target_include_directories(acoustic_pedal PRIVATE
  ${PORTAUDIO_INCLUDE_DIRS}
)

target_link_directories(acoustic_pedal PRIVATE
  ${PORTAUDIO_LIBRARY_DIRS}
)

target_link_libraries(acoustic_pedal PRIVATE
  ${PORTAUDIO_LIBRARIES}
)

target_compile_options(acoustic_pedal PRIVATE
  ${PORTAUDIO_CFLAGS_OTHER}
)

target_link_options(acoustic_pedal PRIVATE
  ${PORTAUDIO_LDFLAGS_OTHER}
)

# Homebrew fallback for Apple Silicon
if(APPLE AND EXISTS "/opt/homebrew/lib/libportaudio.dylib")
  target_link_directories(acoustic_pedal PRIVATE "/opt/homebrew/lib")
  target_link_libraries(acoustic_pedal PRIVATE portaudio)
endif()

# Unit tests
add_executable(run_tests
  tests/test_dsp.cpp
)