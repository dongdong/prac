#include <string>
#include <sstream>
#include <vector>
#include <iostream>

using std::string;
using std::vector;
using std::istringstream;
using std::cin;
using std::cout;
using std::ostream;
using std::endl;

struct Person_info
{
	string name;
	vector<string> phones;
};

void Print(vector<Person_info>& people, ostream& out)
{
	/*
	for (auto& p : people)
	{
		out << "Name: " << p.name << endl;
		out << "Phone: ";
		for (auto& phone : p.phones)
		{
			out << phone << " ";
		}
		cout << endl;
	}
	*/

	for (vector<Person_info>::const_iterator p = people.cbegin(); 
		p != people.cend(); ++p)
	{
		out << "Name: " << p->name << endl;
		out << "Phone: ";
		const vector<string>& ps = p->phones;
		for (vector<string>::const_iterator phone = ps.cbegin(); 
			phone != ps.cend(); ++phone)
		{
			out << *phone << " ";
		}
		cout << endl;
	}
}

int main()
{
	string line, word;
	vector<Person_info> people;

	while (getline(cin, line))
	{
		Person_info info;
		istringstream record(line);
		record >> info.name;
		while (record >> word)
		{
			info.phones.push_back(word);
		}
		people.push_back(info);
	}
	
	Print(people, cout);
}
