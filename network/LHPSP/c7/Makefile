CC=gcc
OBJS=euid_demo test_file
TESTFILE=ROOTFILE

all: $(OBJS)

euid_demo: test_file euid_demo.c
	$(CC) -o euid_demo euid_demo.c

test_file:
	sudo touch $(TESTFILE)
	sudo chmod 600 $(TESTFILE)
	sudo chown root:root $(TESTFILE)

clean:
	rm -rf $(OBJS) $(TESTFILE)
