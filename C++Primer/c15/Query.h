#ifndef __QUERY_H
#define __QUERY_H

#include "TextQuery.h"

class Query {
public:
	Query(const std::string&);
	QueryResult eval(const TextQuery &t) const {
		return q->eval(t);
	}
	std::string rep() const {return q->rep();}
private:
	Query(std::shared_ptr<QueryBase> query) : q(query) {}
	std::shared_ptr<QueryBase> q; 

friend Query operator~(const Query&);
friend Query operator|(const Query&, const Query&);
friend Query operator&(const Query&, const Query&);
};

std::ostream& operator<<(std::ostream& os, const Query& query) {
	return os << query.rep();
}

#endif // __QUERY_H

