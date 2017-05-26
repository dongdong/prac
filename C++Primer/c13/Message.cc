#include "Message.h"
#include "Folder.h"

using namespace std;

Message::Message(const Message& m):
	contents(m.contents), folders(m.folders) {
	add_to_Folders();
}

Message::~Message() {
	remove_from_Folders();
}

Message& Message::operator=(const Message& rhs) {
	remove_from_Folders();
	contents = rhs.contents;
	folders = rhs.folders;
	add_to_Folders();
	return *this;
}

void Message::save(Folder& f) {
	folders.insert(&f);
	f.addMsg(this);
}

void Message::remove(Folder& f) {
	folders.erase(&f);
	f.remMsg(this);
}

void Message::add_to_Folders() {
	for (auto f : folders) {
		f->addMsg(this);
	}
}

void Message::remove_from_Folders() {
	for (auto f : folders) {
		f->remMsg(this);
	}
}

void Message::print(ostream& os) {
	os	<< "Message: " << contents << endl
		<< "in Folders: ";
	for (auto f : folders) {
		os << f->name << "\t";
	}
	os	<< endl;
	
}

void swap(Message& lhs, Message& rhs) {
	using std::swap;
	for (auto f : lhs.folders) {
		f->remMsg(&lhs);
	} 
	for (auto f : rhs.folders) {
		f->remMsg(&rhs);
	}
	
	swap(lhs.folders, rhs.folders);
	swap(lhs.contents, rhs.contents);

	for (auto f : lhs.folders) {
		f->addMsg(&lhs);
	}
	for (auto f : rhs.folders) {
		f->addMsg(&rhs);
	}
}

