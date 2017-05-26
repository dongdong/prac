#ifndef __TEXT_QUERY
#define __TEXT_QUERY

#include <string>
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <memory>

class QueryResult;

class TextQuery {
public:
	using LineNO = std::vector<std::string>::size_type;
	TextQuery(std::istream&);
	QueryResult query(const std::string&) const;
private:
	std::shared_ptr<std::vector<std::string>> contents;
	std::map<std::string, std::shared_ptr<std::set<LineNO>>> wordMap;
};


#endif // __TEXT_QUERY
