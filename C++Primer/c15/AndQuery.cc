#include "AndQuery.h"

QueryResult AndQuery::eval(const TextQuery& Text) const {
	auto left = lhs.eval(text);
	auto right = rhs.eval(text);

	auto ret_lines = make_shared<set<LineNO>>();
	set_intersection(left.begin(), left.end(), 
		right.begin(), right.end, inserter(*ret_lines, ret_lines->begin()));
	return QueryResult(rep, ret_lines, left.get_fine());
}
