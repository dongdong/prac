#ifndef __QUERY_RESULT
#define __QUERY_RESULT

#include "TextQuery.h"

#include <string>
#include <set>
#include <map>
#include <memory>
#include <iostream>


class QueryResult {
public:
	QueryResult(std::string q, std::shared_ptr<std::set<TextQuery::LineNO>> l,
		std::shared_ptr<std::vector<std::string>> c):
		query(q), lines(l), contents(c) {};

private:
	std::string query;
	std::shared_ptr<std::set<TextQuery::LineNO>> lines;
	std::shared_ptr<std::vector<std::string>> contents;

friend std::ostream& operator<<(std::ostream&, const QueryResult&);
};


#endif // __QUERY_RESULT
