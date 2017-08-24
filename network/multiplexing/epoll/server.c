#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/epoll.h>
#include <unistd.h>
#include <sys/types.h>

#define IP_ADDR			"127.0.0.1"
#define PORT			8787
#define MAX_SIZE		1024
#define LIST_ENQ		5
#define FD_SIZE			512
#define EPOLL_EVENTS	128

static int create_server_proc(const char* ip, int port)
{
	int listen_fd;
	struct sockaddr_in serv_addr;

	listen_fd = socket(AF_INET, SOCK_STREAM, 0);
	if (listen_fd == -1)
	{
		perror("[SERVER] socket error: ");
		return -1;
	}

	bzero(&serv_addr, sizeof(serv_addr));
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_port = htons(port);
	inet_pton(AF_INET, ip, &serv_addr.sin_addr);

	if (bind(listen_fd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) == -1)
	{
		perror("[SERVER] bind error: ");
		return -1;
	}

	listen(listen_fd, LIST_ENQ);

	return listen_fd;
}

static void add_event(int epoll_fd, int fd, int state)
{
	struct epoll_event ev;
	ev.events = state;
	ev.data.fd = fd;
	epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd, &ev);
}

static void modify_event(int epoll_fd, int fd, int state)
{
	struct epoll_event ev;
	ev.events = state;
	ev.data.fd = fd;
	epoll_ctl(epoll_fd, EPOLL_CTL_MOD, fd, &ev);
}

static void delete_event(int epoll_fd, int fd, int state)
{
	struct epoll_event ev;
	ev.events = state;
	ev.data.fd = fd;
	epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, &ev);
}

static void handle_accept(int epoll_fd, int serv_fd)
{
	int cli_fd;
	struct sockaddr_in  cli_addr;
	socklen_t cli_addr_len = sizeof(cli_addr);

	cli_fd = accept(serv_fd, (struct sockaddr*)&cli_addr, &cli_addr_len);
	if (cli_fd == -1)
	{
		perror("[SERVER] accept error: ");
		return;
	}

	printf("[SERVER] accept a new client: %s:%d\n", 
			inet_ntoa(cli_addr.sin_addr), cli_addr.sin_port);

	add_event(epoll_fd, cli_fd, EPOLLIN);
}

static void do_read(int epoll_fd, int fd, char *buf)
{
	int n_read = read(fd, buf, MAX_SIZE);
	if (n_read == -1)
	{
		perror("[SERVER] read error: ");
		close(fd);
		delete_event(epoll_fd, fd, EPOLLIN);
	}
	else if (n_read == 0)
	{
		fprintf(stderr, "[SERVER] client close.\n");
		close(fd);
		delete_event(epoll_fd, fd, EPOLLIN);
	} 
	else
	{
		printf("[SERVER] read msg is: %s\n", buf);
		modify_event(epoll_fd, fd, EPOLLOUT);
	}
}

static void do_write(int epoll_fd, int fd, char *buf)
{
	int n_write = write(fd, buf, strlen(buf) + 1);
	if (n_write == -1)
	{
		perror("[SERVER] write error: ");
		close(fd);
		delete_event(epoll_fd, fd, EPOLLOUT);
		return;
	}

	modify_event(epoll_fd,  fd, EPOLLIN);
	memset(buf, 0, MAX_SIZE);
}

static void handle_events(int epoll_fd, struct epoll_event *events, 
		int num, int serv_fd, char *buf)
{
	int i = 0;
	int fd = 0;

	for (i = 0; i < num; i++)
	{
		fd = events[i].data.fd;
		if (fd == serv_fd && events[i].events & EPOLLIN)
		{
			handle_accept(epoll_fd, serv_fd);
		}
		else if (events[i].events & EPOLLIN)
		{
			do_read(epoll_fd, fd, buf);
		}
		else if (events[i].events & EPOLLOUT)
		{
			do_write(epoll_fd, fd, buf);
		}
		else
		{
		}
	}
}


static void handle_client_proc(int serv_fd)
{
	int epoll_fd;
	struct epoll_event events[EPOLL_EVENTS];
	int ret = 0;
	char buf[MAX_SIZE] = {0};

	epoll_fd = epoll_create(FD_SIZE);
	add_event(epoll_fd, serv_fd, EPOLLIN);

	while (1)
	{
		ret = epoll_wait(epoll_fd, events, EPOLL_EVENTS, -1);	
		handle_events(epoll_fd, events, ret, serv_fd, buf);
	}

	close(epoll_fd);
}

int main(int argc,char *argv[])
{
	int serv_fd = create_server_proc(IP_ADDR, PORT);
	if (serv_fd < 0)
	{
		return -1;
	}

	printf("[SERVER] server has establised, waiting for messages from client.\n");

	handle_client_proc(serv_fd);
	
	return 0;
}
