#include "TextQuery.h"
#include "QueryResult.h"
#include <sstream>

using namespace std;

TextQuery::TextQuery(istream& is): content(new vector<string>) {
	string text;
	while (getline(is, text)) {
		content->push_back(text);
		int n = content->size() - 1;
		istringstream line(text);
		string word;
		while (line >> word) {
			auto& lines = wm[word];
			if (!lines) {
				lines.reset(new set<LineNO>);
			}
			lines->insert(n);
		}		
	}
}

QueryResult TextQuery::query(const string& sought) const {
	static shared_ptr<set<LineNO>> noData(new set<LineNO>);
	auto loc = wm.find(sought);
	if (loc == wm.end()) {
		return QueryResult(sought, noData, content);
	} else {
		return QueryResult(sought, loc->second, content);
	}
}
