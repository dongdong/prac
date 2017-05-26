#ifndef __QUERY_H
#define __QUERY_H

#include <string>
#include <memory>
#include <iostream>

#include "QueryResult.h"
#include "TextQuery.h"
#include "QueryBase.h"
#include "WordQuery.h"


class Query {
public:
	Query(const std::string& s): q(new WordQuery(s)) {};
	QueryResult eval(const TextQuery &t) const {
		return q->eval(t);
	} 
	std::string rep() const {
		return q->rep();
	}
private:
	Query(std::shared_ptr<QueryBase> query) : q(query) {}
	std::shared_ptr<QueryBase> q;
friend Query operator~(const Query&);
friend Query operator|(const Query&, const Query&);
friend Query operator&(const Query&, const Query&);
};

#endif // __QUERY_H
