
CPPFLAG=-std=c++11

INCLUDE=-L.

all: TextQuery

TextQuery: Main.cc NotQuery.cc Query.cc QueryResult.cc TextQuery.cc 
	g++ $(CPPFLAG) $(INCLUDE) -o TextQuery Main.cc NotQuery.cc Query.cc QueryResult.cc TextQuery.cc

clean:
	rm -rf TextQuery 
