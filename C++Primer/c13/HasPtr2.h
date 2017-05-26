#ifndef __HAS_PTR2
#define __HAS_PTR2

#include <string>
#include <iostream>
#include <cctype>

class HasPtr2 {
public:
	HasPtr2(const std::string& s = std::string()):
		ps(new std::string(s)), i(0), use(new std::size_t(1)) {}
	HasPtr2(const HasPtr2& p):
		ps(p.ps), i(p.i), use(p.use) { ++*use; }
	HasPtr2& operator=(const HasPtr2&);
	~HasPtr2();
private:
	std::string *ps;
	int i;
	std::size_t *use;
friend std::ostream& print(std::ostream& os, const HasPtr2& p);
};

#endif // __HAS_PTR2
