CC=gcc
OBJS=select_exception_demo epoll_ltet_demo epoll_oneshot_demo

all: $(OBJS)

select_exception_demo: select_exception_demo.c
	$(CC) -o select_exception_demo select_exception_demo.c

epoll_ltet_demo: epoll_ltet_demo.c
	$(CC) -o epoll_ltet_demo epoll_ltet_demo.c

epoll_oneshot_demo: epoll_oneshot_demo.c
	$(CC) -o epoll_oneshot_demo epoll_oneshot_demo.c

clean:
	rm -rf $(OBJS)
