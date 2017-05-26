
CPPFLAG=-std=c++11

all: HasPtr Message StrVec

HasPtr: HasPtrMain.cc HasPtr1.h HasPtr1.cc HasPtr2.h HasPtr2.cc 
	g++ $(CPPFLAG) -o HasPtr HasPtrMain.cc HasPtr1.cc HasPtr2.cc

Message: MessageMain.cc Message.h Message.cc Folder.h Folder.cc
	g++ $(CPPFLAG) -o Message MessageMain.cc Message.cc Folder.cc

StrVec: StrVecMain.cc StrVec.h StrVec.cc
	g++ ${CPPFLAG} -o StrVec StrVecMain.cc StrVec.cc

clean:
	rm -rf HasPtr Message StrVec
