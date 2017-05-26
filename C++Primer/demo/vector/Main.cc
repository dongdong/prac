#include "TVector.h"

#include <string>
#include <iostream>

using namespace std;

int main(int argc, char** argv) {
	
	TVector<string> tvs1;
	tvs1.push_back("s1");
	tvs1.push_back("s2");
	tvs1.push_back("s3");
	cout << "tvs1:\t" << tvs1 << endl;

	TVector<int> tvi1;
	tvi1.push_back(1);
	tvi1.push_back(2);
	tvi1.push_back(3);
	tvi1.push_back(4);
	tvi1.push_back(5);
	cout << "tvi1:\t" << tvi1 << endl;

	TVector<string> tvs2 = tvs1;
	tvs2.push_back("s4");
	cout << "tvs2:\t" << tvs2 << endl;

	TVector<string> tvs3;
	tvs3 = tvs2;
	cout << "tvs3:\t" << tvs3 << endl;

	tvs3 = {"ss1", "ss2", "ss3", "ss4"};
	cout << "tvs3:\t" << tvs3 << endl;

	TVector<string> tvs4 = {"ss11", "ss22", "ss33", "ss44"};
	cout << "tvs4:\t" << tvs4 << endl;

	TVector<string> tvs5(tvs4);
	tvs5.push_back("ss5");
	cout << "tvs5:\t" << tvs5 << endl;
}


