CXX := clang++
CC  := clang

CXXSTD := -std=c++20 
CSTD   := -std=iso9899:2018

CXXFLAGS := -w @make.inc.rsp @make.def.rsp $(CXXSTD)
CFLAGS   := -w @make.inc.rsp @make.def.rsp $(CSTD)

DEBUG_FLAGS   :=  @make.debug.rsp
RELEASE_FLAGS :=  @make.run.rsp

ifdef BUILD
    ifeq ($(BUILD),debug)
        CXXFLAGS += $(DEBUG_FLAGS)
        CFLAGS += $(DEBUG_FLAGS)
    else ifeq ($(BUILD),release)
        CXXFLAGS += $(RELEASE_FLAGS)
        CFLAGS += $(RELEASE_FLAGS)
    endif
else # default to debug build
        CXXFLAGS += $(DEBUG_FLAGS)
        CFLAGS += $(DEBUG_FLAGS)
endif

LLAMA_SOURCES := \
    ./llama.cpp/src/llama.cpp \
    ./llama.cpp/src/llama-sampling.cpp \
    ./llama.cpp/src/llama-vocab.cpp \
    ./llama.cpp/src/llama-grammar.cpp \
    ./llama.cpp/src/unicode.cpp \
    ./llama.cpp/src/unicode-data.cpp

GGML_C_SOURCES := \
    ./llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c \
    ./llama.cpp/ggml/src/ggml-aarch64.c \
    ./llama.cpp/ggml/src/ggml-alloc.c \
    ./llama.cpp/ggml/src/ggml-quants.c \
    ./llama.cpp/ggml/src/ggml.c \
    ./llama.cpp/ggml/src/ggml-cpu/ggml-cpu-aarch64.c \
    ./llama.cpp/ggml/src/ggml-cpu/ggml-cpu-quants.c

GGML_CPP_SOURCES := \
    ./llama.cpp/ggml/src/ggml-backend-reg.cpp \
    ./llama.cpp/ggml/src/ggml-backend.cpp \
    ./llama.cpp/ggml/src/ggml-threading.cpp \
    ./llama.cpp/ggml/src/ggml-cpu/cpu-feats-x86.cpp

GGML_CPU_CPP_SOURCE := ./llama.cpp/ggml/src/ggml-cpu/ggml-cpu.cpp

GGML_CPU_CPP_OBJECT := $(GGML_CPU_CPP_SOURCE:.cpp=-cpp.o)

LLAMA_COMMON_SOURCES := \
    ./llama.cpp/common/arg.cpp \
    ./llama.cpp/common/common.cpp \
    ./llama.cpp/common/console.cpp \
    ./llama.cpp/common/json-schema-to-grammar.cpp \
    ./llama.cpp/common/log.cpp \
    ./llama.cpp/common/ngram-cache.cpp \
    ./llama.cpp/common/sampling.cpp \
    ./llama.cpp/common/speculative.cpp

MAIN_SOURCE := src/main.cpp

LLAMA_OBJECTS        := $(LLAMA_SOURCES:.cpp=.o)
GGML_C_OBJECTS       := $(GGML_C_SOURCES:.c=.o)
GGML_CPP_OBJECTS     := $(GGML_CPP_SOURCES:.cpp=.o)
LLAMA_COMMON_OBJECTS :=  $(LLAMA_COMMON_SOURCES:.cpp=.o)
MAIN_OBJECTS         := $(MAIN_SOURCE:.cpp=.o)

OBJECTS := $(MAIN_OBJECTS) $(LLAMA_OBJECTS) $(GGML_C_OBJECTS) $(GGML_CPP_OBJECTS) $(GGML_CPU_CPP_OBJECT) $(LLAMA_COMMON_OBJECTS)

all: main

main: $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@ -lpthread -ldl

$(GGML_CPU_CPP_OBJECT): $(GGML_CPU_CPP_SOURCE)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(LLAMA_OBJECTS): %.o : %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(LLAMA_COMMON_OBJECTS): %.o : %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) t
