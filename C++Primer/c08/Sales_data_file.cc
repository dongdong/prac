#include "Sales_data.h"
#include <iostream>
#include <fstream>

using std::cin;
using std::cout;
using std::cerr;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::istream;
using std::ostream;

void Process_Sales_data(istream& input, ostream& output)
{
	Sales_data total;
	//if (read(input, total)) 
	if (input >> total)
	{
		Sales_data trans;
		//while (read(input, trans))
		while (input >> trans)
		{
			if (total.isbn() == trans.isbn())
			{
				//total.combine(trans);
				total += trans;
			}
			else
			{
				//print(output, total) << endl;
				output << total << endl;
				total = trans;
			}
		}
		//print(output, total) << endl;
		output << total << endl; 
	} 
	else
	{
		cerr << "No data?!" << endl;
	}
}


int main(int argc, char** argv)
{
	if (argc != 3)
	{
		cerr << "Usage: Sales_data_file infile outfile" << endl;
		return 1;
	}

	ifstream input(argv[1]);
	ofstream output(argv[2]);
	//ofstream output(argv[2], ofstream::app);

	Process_Sales_data(input, output);
	
	return 0;
}
