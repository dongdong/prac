#include "StrVec.h"
#include <iostream>

using namespace std;

void PrintStrVec(const string& name, const StrVec& v) {
	/*
	cout << "StrVec: " << name 
		 << ", size: " << v.size() 
		 << ", capacity: "<< v.capacity() 
		 << endl;
	for (auto b = v.begin(), e = v.end(); b != e; ++b) {
		cout << "\t" << *b << endl;
	}
	*/
	cout << "StrVec: " << name << endl;
	cout << v << endl;
}

int main(int argc, char** argv) {
	StrVec v1;
	v1.push_back("s1");
	v1.push_back("s2");
	v1.push_back("s3");
	PrintStrVec("v1", v1);
	{
		StrVec v2(v1);
		v2.push_back("s4");
		PrintStrVec("v2", v2);
		v2 = v1;
		v1.push_back("s5");
		v2.push_back("s6");
		PrintStrVec("v1", v1);
		PrintStrVec("v2", v2);
	}
	v1.push_back("s7");
	v1.push_back("s8");
	PrintStrVec("v1", v1);

	StrVec v3;
	v3 = {"ss1", "ss2", "ss3"};
	PrintStrVec("v3", v3);
	v3[0] = "sss0";
	cout << "v3 --> " << v3[0] << ", " << v3[1] << endl;
}
