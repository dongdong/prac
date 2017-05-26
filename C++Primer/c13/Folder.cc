#include "Folder.h"
#include "Message.h"

using namespace std;

Folder::~Folder() {
	for (auto m : messages) {
		m->folders.erase(this);
	}	
}

void Folder::addMsg(Message* pm) {
	messages.insert(pm);
}

void Folder::remMsg(Message* pm) {
	messages.erase(pm);
} 

void Folder::print(ostream& os) {
	os	<< "Folder: " << name << endl
		<< "has Messages: " << endl;
	for (auto m : messages) {
		os << "\t" << m->contents << endl;
	}
}	
