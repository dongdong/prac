#ifndef __WORD_QUERY
#define __WORD_QUERY

#include "QueryBase.h"

class WordQuery : public QueryBase {
private:
	WordQuery(const std::string s): queryWord(s) {}
	QueryResult eval(const TextQuery& t) const {
		return t.query(queryWord);
	}
	std::string rep() const { return queryWord; }
	std::string queryWord;

friend class Query;
};

#endif // __WORD_QUERY
