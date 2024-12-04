.PHONY: clean run

# Compiler
CC = nvcc

# Compiler flags
CFLAGS = -g -G

# Target
TARGET = $(basename $(SRC))

# Compile the CUDA file
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

# run
run: $(TARGET)
	./$(TARGET)

# Clean
clean:
	rm -f $(TARGET) *.o *.ptx