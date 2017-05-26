#include "HasPtr1.h"

using std::string;
using std::endl;
using std::ostream;
using std::cout;

/*
HasPtr1& HasPtr1::operator=(const HasPtr1& rhs) {
	cout << "using operator= with const" << endl;
	auto newp = new string(*rhs.ps);
	delete ps;
	ps = newp;
	i = rhs.i;
	return *this;
}
*/

HasPtr1& HasPtr1::operator=(HasPtr1 rhs) {
	cout << "using operator= with no const" << endl;
	swap(*this, rhs);
	return *this;
}

ostream& print(ostream& os, const HasPtr1& p) {
	os	<< "object: " << &p
		<< "\tps: " << *p.ps 
		<< "\taddress: " << p.ps << endl;
	return os;
}
