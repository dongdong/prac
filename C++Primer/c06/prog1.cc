#include <iostream>
#include <string>
#include <cctype>
#include <vector>

using std::string;
using std::cout;
using std::cin;
using std::endl;
using std::vector;
using std::begin;
using std::end;

constexpr size_t rowCnt = 3, colCnt = 4;

//typedef int (*ArrayPointer)[colCnt];
using ArrayPointer = int (*)[colCnt];
//using ArrayPointer = int[][colCnt];
//using ArrayPointer = int[][]; //ERROR
//using ArrayPointer = int**; //ERROR

using In_Array = int[colCnt];

//void PrintArray1(ArrayPointer arr, size_t rowSize)
void PrintArray1(In_Array* arr, size_t rowSize)
{
	for (size_t i = 0; i != rowSize; ++i)
	{
		In_Array &row = *(arr + i);
		for (auto col : row)
		{
			cout << col << " ";
		}
		cout << endl;
	}
}

void PrintArray(ArrayPointer arr, size_t rowSize)
{
	for (size_t i = 0; i != rowSize; ++i)
	{
		for (size_t j = 0; j != colCnt; j++)
		{
			cout << arr[i][j] << " ";
		}		
		cout << endl;
	}	
}

void DemoMulArray()
{
	int arr[rowCnt][colCnt];
	int val = 0;
	
	for (auto &row : arr)
	{
		for (auto &col : row)
		{
			col = val;
			val++;
		}
	} 

	//PrintArray(arr, rowCnt);
	PrintArray1(arr, rowCnt);
}

char &get_val(string& str, string::size_type ix)
{
	return str[ix];
} 

void DemoRefReturn()
{
	string s("a value");
	cout << s << endl;
	get_val(s, 0) = 'A';
	cout << s << endl;
}

const string& shorterString(const string &s1, const string &s2)
{
	cout << "const shorterString" << endl;
	return s1.size() <= s2.size() ? s1 : s2;
} 

string& shorterString(string &s1, string &s2)
{
	cout << "no const shorterString" << endl;
	auto& r = shorterString(const_cast<const string&>(s1),
							const_cast<const string&>(s2));
	return const_cast<string&>(r);
}

void DemoConstOverload()
{
	const string cs1("const hello world");
	const string cs2("const c++ primer");
	
	string s1("hello world");
	string s2("c++ primer");

	shorterString(cs1, cs2);
	shorterString(s1, s2);
}

bool lengthCompare(const string &s1, const string &s2)
{
	return s1.length() >= s2.length();
}

/*
// 1. Use function type
void useBigger(const string &s1, const string &s2,
	bool pf(const string &, const string &))
*/
/*
// 2. Use function pointer
void useBigger(const string &s1, const string &s2,
	bool (*pf)(const string &, const string &))
*/
// 3. typedef function type
//typedef bool Func(const string&, const string&);
// 4. typedef function pointer type
//typedef bool (*Func)(const string&, const string&);
// 5. typedef decltype function type
//typedef decltype(lengthCompare) Func;
// 6. typedef decltypde function pointer
typedef decltype(lengthCompare) *Func;

void useBigger(const string &s1, const string &s2, Func pf)
{
	if (pf(s1, s2))
	{
		cout << s1;
	}
	else 
	{
		cout << s2;
	}
	
	cout << " is Bigger!" << endl;
}

void DemoFunctionPointer()
{
	const string cs1("const hello world");
	const string cs2("const c++ primer");
	
	useBigger(cs1, cs2, lengthCompare);
}

int main()
{
	//DemoMulArray();
	//DemoRefReturn();
	//DemoConstOverload();
	DemoFunctionPointer();
	return 0;
}
