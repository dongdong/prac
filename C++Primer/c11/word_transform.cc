#include <iostream>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <map>

using namespace std;

void buildMap(istream& mapContent, map<string, string>& transMap)
{
	string key, value;
	while (mapContent >> key && getline(mapContent, value))
	{
		if (value.size() > 1)
		{
			transMap[key] = value.substr(1);
		}
		else
		{
			throw runtime_error("no rule for " + key);
		}
	}
}

const string& transform(const string& s, const map<string, string>& m)
{
	auto mapIt = m.find(s);
	if (mapIt != m.cend())
	{
		return mapIt->second;
	} 
	else
	{
		return s;
	}
} 

void wordTransform(istream& mapContent, istream& input)
{
	map<string, string> transMap;
	buildMap(mapContent, transMap);
	string text;
	while (getline(input, text))
	{
		istringstream stream(text);
		string word;
		bool firstWord = true;
		while (stream >> word)
		{
			if (firstWord)
			{
				firstWord = false;
			} 
			else
			{
				cout << " ";
			}
			cout << transform(word, transMap);
		}
		cout << endl; 
	}
}

void DemoWordTransform()
{
	ifstream mapFile("transform_map_file");
	ifstream inputFile("transform_input");
	wordTransform(mapFile, inputFile);	
}


int main()
{
	DemoWordTransform();
	return 0;
}
