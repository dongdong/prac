
CPPFLAG=-std=c++11

all: TextQuery

TextQuery: Main.cc QueryResult.h QueryResult.cc TextQuery.h TextQuery.cc 
	g++ $(CPPFLAG) -o TextQuery Main.cc QueryResult.cc TextQuery.cc

clean:
	rm -rf TextQuery 
