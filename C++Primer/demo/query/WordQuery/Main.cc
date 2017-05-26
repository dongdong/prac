#include "TextQuery.h"
#include "QueryResult.h"
#include "Query.h"
#include "WordQuery.h"

#include <iostream>
#include <fstream>

using namespace std;

void runQueries(istream& input) {
	TextQuery tq(input);
	/*
	while (true) {
		cout << "Enter word to look for, or q to quit: ";
		string s;
		if (!(cin >> s) || s == "q") {
			break;
		}
		//cout << tq.query(s) << endl;
		WordQuery wq(s);
		cout << wq.eval(tq) << endl;
	}
	*/
	//test wordQuery
	Query q("Daddy");
	cout << q.eval(tq) << endl;
	
	Query q1 = ~Query("Alice");
	cout << q1.eval(tq) << endl;
	
}

int main(int argc, char** argv) {
	if (argc != 2) {
		cout << "Usage: TextQuery infile" << endl;
	}
	
	ifstream infile(argv[1]);
	runQueries(infile);

	return 0;
}
