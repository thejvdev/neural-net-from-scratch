CC = clang
CFLAGS = -std=c11 -Wall -Wextra -Wpedantic -O2 -Iinclude

BUILD = build
SRC = src
LIB = lib

SRCS = $(wildcard $(SRC)/*.c)
OBJS = $(patsubst $(SRC)/%.c, $(BUILD)/%.o, $(SRCS))

TARGET = $(LIB)/libsynapse.a

all: $(TARGET)

$(TARGET): $(OBJS) | $(BUILD) $(LIB)
	ar rcs $@ $^

$(BUILD)/%.o: $(SRC)/%.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB):
	mkdir -p $@

$(BUILD):
	mkdir -p $@

clean:
	rm -rf $(LIB) $(BUILD)
