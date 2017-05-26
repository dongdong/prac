#ifndef __AND_QUERY
#define __AND_QUERY

class AndQuery: public BinaryQuery {
private:
	AndQuery(const Query& left, const Query& right):
		BinaryQuery(left, right, "&") {}
	QueryResult eval(const TextQuery&) const;
friend Query operator&(const Query&, const Query&);
};

inline Query operator&(const Query& lhs, const Query& rhs) {
	return std::shared_ptr<QueryBase>(new AndQUery(lhs, rhs));
}

#endif // __AND_QUERY
