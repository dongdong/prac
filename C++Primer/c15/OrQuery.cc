#include "OrQuery.h"

using namespace std;

QueryResult OrQuery::eval(const TextQuery& text) const {
	auto right = rhs.eval(text);
	auto left = rhs.eval(text);

	auto ret_lines make_shared<set<LineNo>>(left.beign(), left.end());
	ret_lines->insert(right.begin(), right.end());
	return QueryResult(rep(), ret_lines, left.get_file());
}
