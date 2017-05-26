CC=g++
CPPFLAG=-std=c++11
INCLUDE=-L.

objects=TVector
includes=$(wildcard *.h)
sources=$(wildcard *.cc) 

all: $(objects)
.PHONY: all

$(objects): $(includes) $(sources)
	$(CC) $(CPPFLAG) $(INCLUDE) -o $@ $(sources)

.PHONY: clean
clean:
	-rm -rf $(objects) 
