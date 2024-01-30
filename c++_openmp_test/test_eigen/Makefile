CC=/opt/homebrew/opt/gcc/bin/g++-13
CFLAGS=-std=c++11 -c -Wall -O3 -I/Users/danis/local/miniconda3/include/eigen3 -I/Users/danis/local/miniconda3/include
LDFLAGS=-llapack -L/opt/homebrew/opt/libomp/lib
CPPFLAGS=-I/opt/homebrew/opt/libomp/include
SOURCES=main.cpp calc_hamK.cpp calc_occupation.cpp calc_exchange.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=xchange.x

all: $(SOURCES) $(EXECUTABLE) functions.h

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@ -lomp

.cpp.o:
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@

clean:
	rm -rf *.o xchange.x

