#ifndef __NOT_QUERY
#define __NOT_QUERY

#include "QueryBase.h"
#include "Query.h"

class NotQuery : public QueryBase {
private:
	NotQuery(const Query &q): query(q) {}
	QueryResult eval(const TextQuery&) const;
	std::string rep() const { 
		return "~(" + query.rep() + ")"; 
	}
	Query query;
friend Query operator~(const Query&);
};


#endif // __NOT_QUERY
