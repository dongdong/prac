#include "Sales_data.h"
#include <iostream>

using std::cin;
using std::cout;
using std::cerr;
using std::endl;

int main(int argc, char** argv)
{
	Sales_data total;
	// if (read(cin, total)) 
	if (cin >> total)
	{
		Sales_data trans;
		// while (read(cin, trans))
		while (cin >> trans)
		{
			if (total.isbn() == trans.isbn())
			{
				//total.combine(trans);
				total += trans;
			}
			else
			{
				//print(cout, total) << endl;
				cout << total << endl;
				total = trans;
			}
		}
		//print(cout, total) << endl;
		cout << total << endl; 
	} 
	else
	{
		cerr << "No data?!" << endl;
	} 

	return 0;
}
