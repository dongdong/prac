#ifndef __QUERY_RESULT
#define __QUERY_RESULT

#include "TextQuery.h"

class QueryResult {
public:
	QueryResult(std::string s, std::shared_ptr<std::set<TextQuery::LineNO>> p,
		std::shared_ptr<std::vector<std::string>> c) : 
		sought(s), lines(p), content(c) {}
private:
	std::string sought;
	std::shared_ptr<std::set<TextQuery::LineNO>> lines;
	std::shared_ptr<std::vector<std::string>> content;

friend std::ostream& print(std::ostream&, const QueryResult&);
};

#endif // __QUERY_RESULT
