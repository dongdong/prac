CC=gcc
OBJS=chat_server_demo
CFLAGS=-g


all: $(OBJS)

chat_server_demo: chat_server_demo.c
	$(CC) $(CFALGS) -o chat_server_demo chat_server_demo.c -lrt

clean:
	rm -rf $(OBJS)
