
CPPFLAG=-std=c++11

all: word_transform word_count

word_transform: word_transform.cc
	g++ $(CPPFLAG) -o word_transform word_transform.cc

word_count: word_count.cc
	g++ $(CPPFLAG) -o word_count word_count.cc

clean:
	rm -rf word_transform word_count 
