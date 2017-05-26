#include <iostream>

#include "Message.h"
#include "Folder.h"

using namespace std;

int main(int argc, char** argv) {
	
	Message m1("m1");
	Message m2("m2");
	Message m3("m3");

	Folder f1("f1");
	Folder f2("f2");

	m1.save(f1);
	m1.save(f2);
	m2.save(f1);
	m3.save(f2);

	m1.print(cout);
	m2.print(cout);
	m3.print(cout);
	f1.print(cout);
	f2.print(cout);

	Message m4(m2);
	Message m5(m3);

	f1.print(cout);
	f2.print(cout);

	m5 = m4;

	f1.print(cout);
	f2.print(cout);

	return 0;
} 
