#ifndef __OR_QUERY
#define __OR_QUERY

#include "BinaryQuery.h"

class OrQuery: public BinaryQuery {
private:
	OrQuery(const Query& left, const Query& right):
		BinaryQuery(left, right, "|") {}
	QueryResult eval(const TextQuery&) const;
friend Query operator|(const Query&, const Query&); 
};

inline Query operator|(const Query& lhs, const Query& rhs) {
	return std::shared_ptr<QueryBase>(new OrQuery(lhs, rhs));
}

#endif // __OR_QUERY
