#include "StrVec.h"
#include <utility>
#include <iostream>

using namespace std;

allocator<string> StrVec::alloc;

StrVec::StrVec(const StrVec &s) {
	auto newdata = alloc_n_copy(s.begin(), s.end());
	elements = newdata.first;
	first_free = cap = newdata.second;
}

StrVec::~StrVec() {
	free();
}

StrVec& StrVec::operator=(const StrVec& rhs) {
	auto data = alloc_n_copy(rhs.begin(), rhs.end());
	free();
	elements = data.first;
	first_free = cap = data.second;
	return *this;
}

StrVec& StrVec::operator=(initializer_list<string> il) {
	auto data = alloc_n_copy(il.begin(), il.end());
	free();
	elements = data.first;
	first_free = cap = data.second;
	return *this;
}

void StrVec::push_back(const string& s) {
	cout << "push_back: " << s << endl;
	chk_n_alloc();
	alloc.construct(first_free++, s);
}

void StrVec::free() {
	cout << "free vec" << endl; 
	if (elements) {
		for (auto p = first_free; p != elements;) {
			alloc.destroy(--p);
		}
		alloc.deallocate(elements, cap - elements);
	}
}

pair<string*, string*> StrVec::alloc_n_copy(const string *b, const string *e) {
	cout << "alloc_n_copy: " << (e - b) << endl;
	auto data = alloc.allocate(e - b);
	return {data, uninitialized_copy(b, e, data)};
}

void StrVec::reallocate() {
	auto newcapacity = ((size() > 0) ? (2 * size()) : 1);
	cout << "reallocate: " << size() << ", " << newcapacity << endl;
	auto newdata = alloc.allocate(newcapacity);
	auto dest = newdata;
	auto elem = elements;
	for (size_t i = 0; i != size(); ++i) {
		alloc.construct(dest++, std::move(*elem++));
	}	
	free();
	elements = newdata;
	first_free = dest;
	cap = elements + newcapacity;
}

ostream& operator<<(ostream& os, const StrVec& v) {
	os  << "size: " << v.size() 
		<< ", capacity: "<< v.capacity() 
		<< endl;
	os  << "contents:\t";
	for (auto b = v.begin(), e = v.end(); b != e; ++b) {
		//cout << "\t" << *b << endl;
		cout << *b << " ";
	}
	return os;
}
