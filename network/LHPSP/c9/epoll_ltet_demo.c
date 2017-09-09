#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/epoll.h>
#include <pthread.h>
#include <libgen.h>
#include <stdbool.h>

#define MAX_EVENT_NUMBER 1024
#define BUFFER_SIZE 10

int set_nonblocking(int fd)
{
	int old_option = fcntl(fd, F_GETFL);
	int new_option = old_option | O_NONBLOCK;
	fcntl(fd, F_SETFL, new_option);
	return old_option;
}

void add_fd(int epollfd, int fd, bool enable_et)
{
	struct epoll_event event;
	event.data.fd = fd;
	event.events = EPOLLIN;
	if (enable_et)
	{
		event.events |= EPOLLET;
	}
	epoll_ctl(epollfd, EPOLL_CTL_ADD, fd, &event);
	set_nonblocking(fd);
}

void lt(struct epoll_event* events, int number, int epollfd, int listenfd)
{
	char buf[BUFFER_SIZE] = {'\0'};
	int i;
	for (i = 0; i < number; i++)
	{
		int sockfd = events[i].data.fd;
		if (sockfd == listenfd)
		{
			struct sockaddr_in client_address;
			socklen_t client_addrlength = sizeof(client_address);
			int connfd = accept(listenfd, (struct sockaddr*)&client_address, 
					&client_addrlength);
			add_fd(epollfd, connfd, false);
		}
		else if (events[i].events & EPOLLIN)
		{
			printf("event trigger once\n");
			memset(buf, '\0', BUFFER_SIZE);
			int ret = recv(sockfd, buf, BUFFER_SIZE - 1, 0);
			if (ret <= 0)
			{
				close(sockfd);
				continue;
			}
			printf("get %d bytes of content: %s\n", ret, buf);
		}
		else
		{
			printf("something else happended\n");
		}
	}
}

void et(struct epoll_event* events, int number, int epollfd, int listenfd)
{
	char buf[BUFFER_SIZE] = {'\0'};
	int i;
	for (i = 0; i < number; i++)
	{
		int sockfd = events[i].data.fd;
		if (sockfd == listenfd)
		{
			struct sockaddr_in client_address;
			socklen_t client_addrlength = sizeof(client_address);
			int connfd = accept(listenfd, (struct sockaddr*)&client_address,
					&client_addrlength);
			add_fd(epollfd, connfd, true);
		}
		else if (events[i].events & EPOLLIN)
		{
			printf("event trigger once\n");
			while (1)
			{
				memset(buf, '\0', BUFFER_SIZE);
				int ret = recv(sockfd, buf, BUFFER_SIZE-1, 0);
				if (ret < 0)
				{
					if ((errno == EAGAIN) || (errno == EWOULDBLOCK))
					{
						printf("read later\n");
						break;
					}
					close(sockfd);
					break;
				}
				else if (ret == 0)
				{
					close(sockfd);
				}
				else
				{
					printf("get %d bytes of content: %s\n", ret, buf);
				}
			}
		}
		else
		{
			printf("something else happened\n");
		}
	}
}

int main(int argc, char** argv)
{
	if (argc <= 3)
	{
		printf("usage: %s ip_address port_number lt|et\n", basename(argv[0]));
		return 1;
	}	
	const char* ip = argv[1];
	int port = atoi(argv[2]);
	const char* mode = argv[3];

	bool enable_et = false;
	if (strncmp(mode, "et", 2) == 0)
	{
		printf("ET enabled\n");
		enable_et = true;
	}

	int ret = 0;
	struct sockaddr_in address;
	bzero(&address, sizeof(address));
	address.sin_family = AF_INET;
	inet_pton(AF_INET, ip, &address.sin_addr);
	address.sin_port = htons(port);

	int listenfd = socket(AF_INET, SOCK_STREAM, 0);
	assert(listenfd >= 0);

	ret = bind(listenfd, (struct sockaddr*)&address, sizeof(address));
	assert(ret != -1);
	
	ret = listen(listenfd, 5);
	assert(ret != -1);

	struct epoll_event events[MAX_EVENT_NUMBER];
	int epollfd = epoll_create(5);
	assert(epollfd != -1);
	add_fd(epollfd, listenfd, enable_et);
	
	while (1)
	{
		ret = epoll_wait(epollfd, events, MAX_EVENT_NUMBER, -1);
		if (ret < 0)
		{
			perror("epoll_wait failed:");
			break;
		}
		if (enable_et)
		{
			et(events, ret, epollfd, listenfd);
		}
		else
		{
			lt(events, ret, epollfd, listenfd);
		}
	}
	
	close(listenfd);
	return 0;
}

