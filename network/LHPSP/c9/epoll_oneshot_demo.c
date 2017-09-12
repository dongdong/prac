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
#define BUFFER_SIZE 1024

struct fds
{
	int epollfd;
	int sockfd;
};


int set_nonblocking(int fd)
{
	int old_option = fcntl(fd, F_GETFL);
	int new_option = old_option | O_NONBLOCK;
	fcntl(fd, F_SETFL, new_option);
	return old_option;
}

void add_fd(int epollfd, int fd, bool oneshot)
{
	struct epoll_event event;
	event.data.fd = fd;
	event.events = EPOLLIN | EPOLLET;
	if (oneshot)
	{
		event.events |= EPOLLONESHOT;
	}
	epoll_ctl(epollfd, EPOLL_CTL_ADD, fd, &event);
	set_nonblocking(fd);
}

void reset_oneshot(int epollfd, int fd)
{
	struct epoll_event event;
	event.data.fd = fd;
	event.events = EPOLLIN | EPOLLET | EPOLLONESHOT;
	epoll_ctl(epollfd, EPOLL_CTL_MOD, fd, &event);
}

void* worker(void* arg)
{
	int epollfd = ((struct fds*)arg)->epollfd;
	int sockfd = ((struct fds*)arg)->sockfd;
	printf("[SERVER] start new thread to receive data on fd: %d\n", sockfd);
	char buf[BUFFER_SIZE] = {'\0'};
	
	while (1)
	{
		int ret = recv(sockfd, buf, BUFFER_SIZE - 1, 0);
		if (ret == 0)
		{
			close(sockfd);
			printf("[SERVER] client closed the connection\n");
			break;
		}
		else if (ret < 0)
		{	if (errno == EAGAIN)
			{
				reset_oneshot(epollfd, sockfd);
				printf("[SERVER] read later\n");
			}
			else
			{
				perror("[SERVER] recv failed");
				close(sockfd);
			}
			break;
		}
		else
		{
			printf("[SERVER] get content: %s\n", buf);
			//sleep(1);
		}
	}
	printf("[SERVER] end thread receiving data on fd: %d\n", sockfd);
}

int main(int argc, char** argv)
{	
	if (argc <= 2)
	{
		printf("usage: %s ip_address port_number lt|et\n", basename(argv[0]));
		return 1;
	}	
	const char* ip = argv[1];
	int port = atoi(argv[2]);

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
	add_fd(epollfd, listenfd, false);
	
	while(1)
	{
		int ret = epoll_wait(epollfd, events, MAX_EVENT_NUMBER, -1);
		if (ret < 0)
		{
			perror("epoll_wait failed:");
			break;
		}
		
		int i;
		for (i = 0; i < ret; i++)
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
				pthread_t thread;
				struct fds fds_for_new_worker;
				fds_for_new_worker.epollfd = epollfd;
				fds_for_new_worker.sockfd = sockfd;
				pthread_create(&thread, NULL, worker, (void*)&fds_for_new_worker);
			}
			else
			{
				printf("[SERVER] something else happended\n");
			}
		}		
	}

	close(listenfd);

	return 0;
}

