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

void DemoRangeFor()
{
	string s("Hello, World!!!");
	for (auto &c : s)
	{
		c = toupper(c);
	}
	cout << s << endl;
}

void DemoStringIndex()
{
	string s("some thing");
	//s[0] = toupper(s[0]);
	for (decltype(s.size()) index = 0;
		index != s.size() && !isspace(s[index]);
		index++)
	{
		s[index] = toupper(s[index]);
	}
	cout << s << endl;
}

void DemoHexDigits()
{
	const string hexDigits = "0123456789ABCDEF";
	cout << "Enter a series of number between 0 and 15"
		 << " seperated by space. Hit ENTER when finished:"
		 << endl;
	string result;
	string::size_type n;
	while (cin >> n)
	{
		if (n < hexDigits.size())
		{
			result += hexDigits[n];
			result += ' ';
		}
	}
	
	cout << "Your hex number is " << result << endl;
}

void DemoVector()
{
	vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9};
	for (auto &i : v)
	{
		i *= i;
	}
	for (auto i : v)
	{
		cout << i << " ";
	}
	cout << endl;
}

void DemoScoreRange()
{
	vector<unsigned> scores(11, 0);
	unsigned grade;
	while (cin >> grade)
	{
		if (grade <= 100)
		{
			++scores[grade/10];
		}
	}
	for (auto u : scores)
	{
		cout << u << " ";
	}
	cout << endl;
}

void DemoStringIter()
{
	string s("some thing");
	for (auto it = s.begin(); 
		it != s.end() && !isspace(*it);
		it++)
	{
		*it = toupper(*it);
	}
	cout << s << endl;
}


void DemoMulArray()
{
	constexpr size_t rowCnt = 3, colCnt = 4;
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
	
	for (auto &row : arr)
	{
		for (auto col : row)
		{
			cout << col << " ";
		}
		cout << endl;
	}
	
	for (size_t i = 0; i != rowCnt; i++)
	{
		for (size_t j = 0; j != colCnt; j++)
		{
			cout << arr[i][j] << " ";
		}
		cout << endl;
	}

	for (auto p = arr; p != arr + rowCnt; p++)
	{
		for (auto q = *p; q != *p + colCnt; q++)
		{
			cout << *q << " ";
		}
		cout << endl;
	}
	
	for (auto p = begin(arr); p != end(arr); p++)
	{
		for (auto q = begin(*p); q != end(*p); q++)
		{
			cout << *q << " ";
		}
		cout << endl;
	}

	using in_array = int[colCnt];
	//typedef int in_array[4];

	for (in_array *p = arr; p != arr + rowCnt; p++)
	{
		for (int *q = *p; q != *p + colCnt; q++)
		{
			cout << *q << " ";
		}
		cout << endl;
	}

	//for (int **p = arr; p != arr + rowCnt; p++) //ERROR
	for (int (*p)[colCnt] = arr; p != arr + rowCnt; p++)
	{
		for (int *q = *p; q != *p + colCnt; q++)
		{
			cout << *q << " ";
		}
		cout << endl;
	}
	
	cout << "size: " << sizeof(arr) << endl;
}

void TestCharSize()
{
	const char *s1[] = {
		"hello",
		"world",
		"c++ primer"
	};
	cout << "size:" << sizeof(s1) << endl;
	// size of three pointer
	
	const char s2[3][15] = {
		"hello",
		"world",
		"c++ primer"
	};
	cout << "size:" << sizeof(s2) << endl;
}

int main()
{
	//DemoRangeFor();
	//DemoStringIndex();
	//DemoHexDigits();
	//DemoVector();
	//DemoScoreRange();
	//DemoStringIter();
	//DemoMulArray();
	TestCharSize();
	return 0;
}
