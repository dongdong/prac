#ifndef __TEXT_QUERY
#define __TEXT_QUERY

#include <string>
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <memory>


class QueryResult;

class TextQuery {
public:
	using LineNO = std::vector<std::string>::size_type;
	TextQuery(std::istream&);
	QueryResult query(const std::string&) const;
private:
	std::shared_ptr<std::vector<std::string>> content;
	std::map<std::string, std::shared_ptr<std::set<LineNO>>> wm;
};


#endif // __TEXT_QUERY
