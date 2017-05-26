#ifndef __QUERY_BASE
#define __QUERY_BASE

#include <string>
#include "TextQuery.h"
#include "QueryResult.h"

class QueryBase {
protected:
	using LineNO = TextQuery::LineNO;
	virtual ~QueryBase() = default;
private:
	virtual QueryResult eval(const TextQuery& t) const = 0;
	virtual std::string rep() const = 0;
friend class Query;
};

#endif // __QUERY_BASE
