CC=gcc
OBJS=dup_demo writev_demo

all: $(OBJS)

dup_demo: dup_demo.c
	$(CC) -o dup_demo dup_demo.c

writev_demo: writev_demo.c
	$(CC) -o writev_demo writev_demo.c

clean:
	rm -rf $(OBJS)
