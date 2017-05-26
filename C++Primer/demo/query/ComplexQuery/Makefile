CC=g++
CPPFLAG=-std=c++11
INCLUDE=-L.

objects=TextQuery
includes=$(wildcard *.h)
sources=$(wildcard *.cc) 

all: $(objects)
.PHONY: all

TextQuery: $(includes) $(sources)
	$(CC) $(CPPFLAG) $(INCLUDE) -o $@ $(sources)

.PHONY: clean
clean:
	-rm -rf $(objects) 
