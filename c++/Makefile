CC=clang++
CFLAGS=-std=c++11  -c -Wall -O2 -I/Users/danis/local/miniconda3/include/  
LDFLAGS= -llapack
SOURCES= main.cpp calc_hamK.cpp calc_exchange.cpp calc_occupation.cpp coordination_sort.cpp inverse_matrix.cpp 
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE= xchange.x

all: $(SOURCES) $(EXECUTABLE) functions.h

$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC)  $(CFLAGS) $< -o $@

clean:
	rm -rf *.o xchange.x
