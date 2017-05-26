#ifndef __FOLDER_H
#define __FOLDER_H
#include <string>
#include <set>
#include <iostream>

class Message;

class Folder {
public:
	Folder(const std::string& str = ""):
		name(str) { }
	~Folder();
	void addMsg(Message*);
	void remMsg(Message*);
	void print(std::ostream&);
private:
	std::string name;
	std::set<Message*> messages;
friend class Message;
};

#endif // __FOLDER_H
