#ifndef __BINARY_QUERY
#define __BINARY_QUERY

class BinaryQUery: public QueryBase {
protected:
	BinaryQuery(const Query& l, const Query& r, std::string s) :
		lhs(l), rhs(r), opSym(s) {}
	std::string rep() const;
	Query lhs, rhs;
	std::string opSym;
}

inline std::string rep() const {
	return "(" + lhs.rep() + " "
		+ opSym + " " 
		+ rhs.rep() + ")";
};

#endif // __BINARY_QUERY
