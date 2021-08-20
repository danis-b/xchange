CC=g++
CFLAGS=-std=c++11 -c -Wall -O2 -I/opt/local/include  
LDFLAGS= -llapack
SOURCES= xchange.cpp calc_exchange.cpp calc_occupation.cpp coordination_sort.cpp 
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE= xchange.x

all: $(SOURCES) $(EXECUTABLE) functions.h

$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf *.o xchange.x