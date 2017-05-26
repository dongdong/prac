#include "QueryResult.h"

using namespace std;

static string make_plural(size_t ctr, const string& word, 
	const string& ending) {
	return (ctr > 1) ? word + ending : word; 
}

ostream& operator<<(ostream& os, const QueryResult& qr) {
	os	<< qr.query << " occurs " << qr.lines->size() << " "
		<< make_plural(qr.lines->size(), "time", "s") << endl;
	for (auto num : *qr.lines) {
		os	<< "\tline " << num + 1 << ") "
			<< *(qr.contents->begin() + num) << endl;
	}
	return os;
}
	

