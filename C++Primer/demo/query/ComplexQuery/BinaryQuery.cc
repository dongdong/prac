#include "BinaryQuery.h"

//#include <set>
//#include <memory>

using namespace std;

QueryResult AndQuery::eval(const TextQuery& text) const {
	auto left  = lhs.eval(text);
	auto right = rhs.eval(text);

	auto ret_lines = make_shared<set<LineNO>>();
	set_intersection(left.begin(), left.end(), right.begin(), right.end(), 
		inserter(*ret_lines, ret_lines->begin()));

	return QueryResult(rep(), ret_lines, left.get_file());
}

QueryResult OrQuery::eval(const TextQuery& text) const {
	auto left  = lhs.eval(text);
	auto right = rhs.eval(text);

	auto ret_lines = make_shared<set<LineNO>>(left.begin(), left.end());
	ret_lines->insert(right.begin(), right.end());

	return QueryResult(rep(), ret_lines, left.get_file());
}

Query operator&(const Query& lhs, const Query& rhs) {
	return shared_ptr<QueryBase>(new AndQuery(lhs, rhs));
}

Query operator|(const Query& lhs, const Query& rhs) {
	return shared_ptr<QueryBase>(new OrQuery(lhs, rhs));
}
