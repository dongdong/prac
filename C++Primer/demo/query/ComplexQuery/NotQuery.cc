#include "NotQuery.h"

using namespace std;

QueryResult NotQuery::eval(const TextQuery& t) const {
	auto result = query.eval(t);
	auto ret_lines = make_shared<set<LineNO>>();
	auto beg = result.begin(), end = result.end();
	auto sz = result.get_file()->size();
	for (size_t n = 0; n != sz; ++n) {
		if (beg == end || *beg != n) {
			ret_lines->insert(n);
		} else if (beg != end) {
			++beg;
		}
	}
	return QueryResult(rep(), ret_lines, result.get_file());
}

Query operator~(const Query& operand) {
	return std::shared_ptr<QueryBase>(new NotQuery(operand));
}
