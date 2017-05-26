#include <unordered_map>
#include <string>
#include <fstream>
#include <iostream>
#include <cctype>

using namespace std;

void countWord(istream& content, unordered_map<string, size_t>& wordCount)
{
	string word;
	while (content >> word)
	{
		++wordCount[word];
	}
}

int main(int argc, char** argv)
{
	unordered_map<string, size_t> wordCount;
	
	/*
	ifstream input1("word_transform.cc");
	ifstream input2("word_count.cc");
	
	countWord(input1, wordCount);
	countWord(input2, wordCount);
	*/

	for (int i = 1; i < argc; ++i)
	{
		ifstream input(argv[i]);
		countWord(input, wordCount);
	}	

	for (const auto &w : wordCount)
	{
		cout << w.first << " occurs " << w.second
			 << ((w.second > 1) ? " times" : " time")
			 << endl;
	}

	return 0;
}
