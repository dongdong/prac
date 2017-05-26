#ifndef __T_VECTOR
#define __T_VECTOR

#include <memory>
#include <cctype>
#include <iostream>

template <typename T> class TVector;
template <typename T> 
std::ostream& operator<<(std::ostream& os, const TVector<T>& tv); 

template <typename T>
class TVector {
public:
	TVector(): elements(nullptr), first_free(nullptr), cap(nullptr) {}
	TVector(const TVector<T>&);
	TVector(std::initializer_list<T>);
	TVector<T>& operator=(const TVector<T>&);
	TVector<T>& operator=(std::initializer_list<T>);
	~TVector();
	void push_back(const T&);
	size_t size() const {return first_free - elements;}
	size_t capacity() const {return cap - elements;}
	T* begin() const {return elements;}
	T* end() const {return first_free;}
	T& operator[](std::size_t n) {return elements[n];}	
	const T& operator[](std::size_t n) const {return elements[n];}	
private:
	static std::allocator<T> alloc;
	void chk_n_alloc();
	std::pair<T*, T*> alloc_n_copy(const T*, const T*);
	void free();
	void reallocate();
	T* elements;
	T* first_free;
	T* cap;
friend std::ostream& operator<<<T>(std::ostream&, const TVector<T>&);
};

template <typename T>
std::allocator<T> TVector<T>::alloc;

template <typename T>
TVector<T>::TVector(const TVector<T>& tv) {
	auto newdata = alloc_n_copy(tv.begin(), tv.end());
	elements = newdata.first;
	first_free = cap = newdata.second;
}

template <typename T>
TVector<T>::TVector(std::initializer_list<T> il) {
	auto data = alloc_n_copy(il.begin(), il.end());
	elements = data.first;
	first_free = cap = data.second;
}

template <typename T>
TVector<T>& TVector<T>::operator=(const TVector<T>& tv) {
	auto newdata = alloc_n_copy(tv.begin(), tv.end());
	free();
	elements = newdata.first;
	first_free = cap = newdata.second;
	return *this;
}

template <typename T>
TVector<T>& TVector<T>::operator=(std::initializer_list<T> il) {
	auto data = alloc_n_copy(il.begin(), il.end());
	free();
	elements = data.first;
	first_free = cap = data.second;
	return *this;
}

template <typename T>
TVector<T>::~TVector() {
	free();
}

template <typename T>
void TVector<T>::push_back(const T& t) {
	chk_n_alloc();
	alloc.construct(first_free++, t);
}

template <typename T>
void TVector<T>::free() {
	if (elements) {
		for (auto p = first_free; p != elements;) {
			alloc.destroy(--p);
		}
		alloc.deallocate(elements, cap - elements);
	}
}

template <typename T>
void TVector<T>::chk_n_alloc() {
	if (size() == capacity()) {
		reallocate();
	}
}

template <typename T>
std::pair<T*, T*> TVector<T>::alloc_n_copy(const T* b, const T* e) {
	auto data = alloc.allocate(e - b);
	return {data, uninitialized_copy(b, e, data)};
}

template <typename T>
void TVector<T>::reallocate() {
	auto newcapacity = ((size() > 0) ? (2 * size()) : 1);
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

template <typename T>
std::ostream& operator<<(std::ostream& os, const TVector<T>& tv) {
	std::cout << "size=" << tv.size() 
			  << ", capacity=" << tv.capacity()
			  << ", contents=[ ";
	for (auto b = tv.begin(), e = tv.end(); b != e; ++b) {
		std::cout << *b << " ";
	}
	std::cout << "]";
	return os;
}

#endif // __T_VECTOR
