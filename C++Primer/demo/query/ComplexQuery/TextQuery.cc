#include "TextQuery.h"
#include "QueryResult.h"

#include <sstream>

using namespace std;

static string trim(const string& str) {
	string output = "";
	for (string::size_type i = 0; i != str.size(); ++i) {
		if (!ispunct(str[i])) {
			output += str[i];
		} 
	}
	return output;
}

TextQuery::TextQuery(istream& input): contents(new vector<string>) {
	string line;
	LineNO lineNum = 0;
	while (getline(input, line)) {
		contents->push_back(line);
		istringstream ss(line);
		string word;
		while (ss >> word) {
			string trimedWord = trim(word);
			auto& lines = wordMap[trimedWord];
			if (!lines) {
				lines.reset(new set<LineNO>);
			}
			lines->insert(lineNum);
		}
		++lineNum;
	}
}

QueryResult TextQuery::query(const string& q) const {
	static shared_ptr<set<LineNO>> noData(new set<LineNO>);
	auto loc = wordMap.find(q);
	if (loc == wordMap.end()) {
		return QueryResult(q, noData, contents);
	} else {
		return QueryResult(q, loc->second, contents);
	}
}
