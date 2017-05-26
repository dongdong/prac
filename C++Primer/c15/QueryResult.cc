#include "QueryResult.h"
#include <iostream>
#include <cctype>

using namespace std;

static string make_plural(size_t ctr, const string& word, 
		const string& ending) {
	return (ctr > 1) ? word + ending : word;
}

ostream& print(ostream& os, const QueryResult& qr) {
	os << qr.sought << " occurs " << qr.lines->size() << " "
	   << make_plural(qr.lines->size(), "time", "s") << endl;
	for (auto num : *qr.lines) {
		os << "\t(line " << num + 1 << ") "
		   << *(qr.content->begin() + num) << endl; 
	} 
	return os;
}
