#include "TextQuery.h"
#include "QueryResult.h"

#include <fstream>

using namespace std;


void runQueries(istream& input) {
	TextQuery tq(input);
	while (true) {
		cout << "enter word to look for, or q to quit: ";
		string s;
		if (!(cin >> s) || s == "q") {
			break;
		}
		print(cout, tq.query(s)) << endl;
	}
}

int main(int argc, char** argv) {
	if (argc != 2) {
		cout << "Usage: TextQuery infile" << endl;
		return 1;
	}

	ifstream infile(argv[1]);
	runQueries(infile);
	
	return 0;
}
