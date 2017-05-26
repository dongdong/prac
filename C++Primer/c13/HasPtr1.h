#ifndef __HAS_PTR1
#define __HAS_PTR1

#include <string>
#include <iostream>

class HasPtr1 {
public:
	HasPtr1(const std::string& s = std::string()):
		ps(new std::string(s)), i(0) {}
	HasPtr1(const HasPtr1& p):
		ps(new std::string(*p.ps)), i(p.i) {}
	//HasPtr1& operator=(const HasPtr1&);
	HasPtr1& operator=(HasPtr1);
	~HasPtr1() {delete ps;}
private:
	std::string *ps;
	int i;
friend std::ostream& print(std::ostream& os, const HasPtr1& p);
friend void swap(HasPtr1&, HasPtr1&);
};

inline void swap(HasPtr1& lhs, HasPtr1& rhs) {
	using std::swap;
	swap(lhs.ps, rhs.ps);
	swap(lhs.i, rhs.i);
} 

#endif // __HAS_PTR1
