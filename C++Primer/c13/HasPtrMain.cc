#include "HasPtr1.h"
#include "HasPtr2.h"

using namespace std;

int main()
{
	HasPtr1 p1("value");
	HasPtr1 p2(p1);
	HasPtr1 p3 = p2;	
	HasPtr1 p4;
	p4 = p3;

	print(cout, p1);
	print(cout, p2);
	print(cout, p3);
	print(cout, p4);
	
	HasPtr2 p5("pointer");
	print(cout, p5);
	do {
		HasPtr2 p6(p5);
		print(cout, p6);
		HasPtr2 p7 = p5;
		print(cout, p7);
		print(cout, p5);
	} while (0);

	print(cout, p5);
	
	HasPtr2 p8;
	p8 = p5;
	print(cout, p8);
	print(cout, p5);
}
