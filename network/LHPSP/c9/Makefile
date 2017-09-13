CC=gcc -g 
OBJS=select_exception_demo epoll_ltet_demo epoll_oneshot_demo nonblock_connect_demo chat_client chat_server echo_server

all: $(OBJS)

select_exception_demo: select_exception_demo.c
	$(CC) -o select_exception_demo select_exception_demo.c

epoll_ltet_demo: epoll_ltet_demo.c
	$(CC) -o epoll_ltet_demo epoll_ltet_demo.c

epoll_oneshot_demo: epoll_oneshot_demo.c
	$(CC) -o epoll_oneshot_demo epoll_oneshot_demo.c -lpthread

nonblock_connect_demo: nonblock_connect_demo.c
	$(CC) -o nonblock_connect_demo nonblock_connect_demo.c

chat_client: chat_client.c
	$(CC) -o chat_client chat_client.c

chat_server: chat_server.c
	$(CC) -o chat_server chat_server.c

echo_server: echo_server.c
	$(CC) -o echo_server echo_server.c

clean:
	rm -rf $(OBJS)
