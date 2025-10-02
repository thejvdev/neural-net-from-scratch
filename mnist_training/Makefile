CC = clang
CFLAGS = -std=c11 -Wall -Wextra -I../synapse/include
LDFLAGS = -L../synapse/lib
LDLIBS = -lsynapse

TARGET = train

all: $(TARGET)

$(TARGET): main.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

clean:
	rm -f $(TARGET) *.o
