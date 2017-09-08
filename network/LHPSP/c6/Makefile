CC=gcc
OBJS=dup_demo writev_demo sendfile_demo splice_demo tee_demo

all: $(OBJS)

dup_demo: dup_demo.c
	$(CC) -o dup_demo dup_demo.c

writev_demo: writev_demo.c
	$(CC) -o writev_demo writev_demo.c

sendfile_demo: sendfile_demo.c
	$(CC) -o sendfile_demo sendfile_demo.c

splice_demo: splice_demo.c
	$(CC) -o splice_demo splice_demo.c
	
tee_demo: tee_demo.c
	$(CC) -o tee_demo tee_demo.c

clean:
	rm -rf $(OBJS)
