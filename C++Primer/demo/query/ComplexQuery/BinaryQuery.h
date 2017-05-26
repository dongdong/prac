#ifndef __BINARY_QUERY
#define __BINARY_QUERY

#include "QueryBase.h"
#include "Query.h"

class BinaryQuery: public QueryBase {
protected:
	BinaryQuery(const Query& l, const Query& r, std::string s):
		lhs(l), rhs(r), opSym(s) {}
	std::string rep() const {
		return "(" + lhs.rep() + " " + opSym + " "
			+ rhs.rep() + ")";
	}
	Query lhs, rhs;
	std::string opSym;
};

class AndQuery: public BinaryQuery {
private:
	AndQuery(const Query& l, const Query& r):
		BinaryQuery(l, r, "&") {}
	QueryResult eval(const TextQuery&) const;
friend Query operator&(const Query&, const Query&);
};

class OrQuery: public BinaryQuery {
private:
	OrQuery(const Query& l, const Query& r):
		BinaryQuery(l, r, "|") {}
	QueryResult eval(const TextQuery&) const;
friend Query operator|(const Query&, const Query&);
};

#endif // __BINARY_QUERY
