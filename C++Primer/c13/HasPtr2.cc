#include "HasPtr2.h"

using std::string;
using std::endl;
using std::ostream;

HasPtr2::~HasPtr2() {
	if (--*use == 0) {
		delete ps;
		delete use;
	}
}

HasPtr2& HasPtr2::operator=(const HasPtr2& rhs) {
	++*rhs.use;
	if (--*use == 0) {
		delete ps;
		delete use;
	}
	ps = rhs.ps;
	i = rhs.i;
	use = rhs.use;
	return *this;
}


ostream& print(ostream& os, const HasPtr2& p) {
	os	<< "object: " << &p
		<< "\tps: " << *p.ps 
		<< "\taddress: " << p.ps 
		<< "\tuse: " << *p.use
		<< endl;
	return os;
}

