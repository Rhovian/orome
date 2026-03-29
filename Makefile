CC = clang
CFLAGS = -O2 -Wall -Wextra -fobjc-arc -Iinference/include -Iinference/vendor
FRAMEWORKS = -framework Metal -framework Foundation
LDFLAGS =

# Multi-file build
OROME_TARGET = orome
OROME_SRCS = inference/cli/main.m inference/core/engine.m inference/core/metal.m \
             inference/core/kernels.m inference/core/tokenizer.m inference/server/server.m \
             inference/core/gguf.m inference/core/format.m \
             inference/core/engine_qwen35_hybrid.m
OROME_OBJS = $(OROME_SRCS:.m=.o)

# Metal shaders
METALC = xcrun -sdk macosx metal
METALLIB_TOOL = xcrun -sdk macosx metallib
SHADER_SRC = inference/shaders/shaders.metal
SHADER_AIR = inference/shaders/shaders.air
SHADER_LIB = inference/shaders/shaders.metallib

.PHONY: all clean bench

all: $(SHADER_LIB) $(OROME_TARGET)

%.o: %.m inference/include/orome.h
	$(CC) $(CFLAGS) -c $< -o $@

$(OROME_TARGET): $(OROME_OBJS)
	$(CC) $(CFLAGS) $(FRAMEWORKS) $(LDFLAGS) $(OROME_OBJS) -o $(OROME_TARGET)
	@echo "Built: $(OROME_TARGET)"

metallib: $(SHADER_LIB)

$(SHADER_AIR): $(SHADER_SRC)
	$(METALC) -c -ffast-math $(SHADER_SRC) -o $(SHADER_AIR)

$(SHADER_LIB): $(SHADER_AIR)
	$(METALLIB_TOOL) $(SHADER_AIR) -o $(SHADER_LIB)

clean:
	rm -f $(OROME_TARGET) $(OROME_OBJS) $(SHADER_AIR) $(SHADER_LIB)

bench: $(OROME_TARGET)
	python3 inference/tools/benchmark.py --trials 1 --json

MODEL ?= /Users/j/Code/lllm/models/Qwen3.5-35B-A3B-Q4_K_S.gguf
PORT ?= 8080

serve: $(OROME_TARGET)
	./$(OROME_TARGET) --model $(MODEL) --serve $(PORT)

chat:
	@python3 inference/tools/chat.py --port $(PORT)
