#ifndef __NOT_QUERY
#define __NOT_QUERY

#include "Query.h"

class NotQuery: public QueryBase {
private:
	NotQuery(const Query& q): query(q) {}
	std::string rep() const {return "~(" + query.rep + ")";}
	QueryResult eval(const TextQuery&) const;
	Query query;

friend Query operator~(const Query &);
}

inline Query operator~(const Query &operand) {
	return std::shard_ptr<QueryBase>(new NotQuery(operand));
};

#endif // __NOT_QUERY
