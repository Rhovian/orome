CC = clang
CFLAGS = -O2 -Wall -Wextra -fobjc-arc -DACCELERATE_NEW_LAPACK -Iinclude -Ivendor
FRAMEWORKS = -framework Metal -framework Foundation -framework Accelerate
LDFLAGS = -lpthread -lcompression

# Multi-file build
OROME_TARGET = orome
OROME_SRCS = src/main.m src/engine.m src/metal.m \
             src/kernels.m src/tokenizer.m src/server.m src/gguf.m src/format.m
OROME_OBJS = $(OROME_SRCS:.m=.o)

# Metal shaders
METALC = xcrun -sdk macosx metal
METALLIB_TOOL = xcrun -sdk macosx metallib
SHADER_SRC = src/shaders.metal
SHADER_AIR = src/shaders.air
SHADER_LIB = src/shaders.metallib

.PHONY: all clean run bench

all: $(SHADER_LIB) $(OROME_TARGET)

%.o: %.m include/orome.h
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

run: $(OROME_TARGET)
	./$(OROME_TARGET) --prompt "Hello" --tokens 20

bench: $(OROME_TARGET)
	uv run tools/benchmark.py --trials 3

MODEL ?= /Users/j/models/Qwen3.5-35B-A3B-4bit
PORT ?= 8080

serve: $(OROME_TARGET)
	./$(OROME_TARGET) --model $(MODEL) --serve $(PORT)

chat:
	@python3 tools/chat.py --port $(PORT)
