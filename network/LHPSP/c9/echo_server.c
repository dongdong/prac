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
#include <libgen.h>
#include <sys/epoll.h>

#define MAX_EVENT_NUMBER 1024
#define TCP_BUFFER_SIZE 512
#define UDP_BUFFER_SIZE 1024

int set_nonblocking(int fd)
{
	int old_option = fcntl(fd, F_GETFL);
	int new_option = old_option | O_NONBLOCK;
	fcntl(fd, F_SETFL, new_option);
	return old_option;
}

void add_fd(int epollfd, int fd)
{
	struct epoll_event event;
	event.data.fd = fd;
	event.events = EPOLLIN | EPOLLET;
	epoll_ctl(epollfd, EPOLL_CTL_ADD, fd, &event);
	set_nonblocking(fd);
}

int main(int argc, char** argv)
{
	if (argc <= 2)
	{
		printf("usage: %s ip_address port_number\n", basename(argv[0]));
		return 1;
	}
	const char* ip = argv[1];
	int port = atoi(argv[2]);
	
	int ret = 0;
	struct sockaddr_in address;
	
	// TCP
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

	// UDP
	bzero(&address, sizeof(address));
	address.sin_family = AF_INET;
	inet_pton(AF_INET, ip, &address.sin_addr);
	address.sin_port = htons(port);
	
	int udpfd = socket(AF_INET, SOCK_DGRAM, 0);
	assert(udpfd >= 0);
	ret = bind(udpfd, (struct sockaddr*)&address, sizeof(address));
	assert(ret != -1);
	
	struct epoll_event events[MAX_EVENT_NUMBER];
	int epollfd = epoll_create(5);
	assert(epollfd != -1);
	add_fd(epollfd, listenfd);
	add_fd(epollfd, udpfd);

	while (1)
	{
		int number = epoll_wait(epollfd, events, MAX_EVENT_NUMBER, -1);
		if (number < 0)
		{
			perror("[SERVER] epoll failed");
			break;
		}
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
				add_fd(epollfd, connfd);
			}
			else if (sockfd == udpfd)
			{
				char buf[UDP_BUFFER_SIZE] = {'\0'};
				struct sockaddr_in client_address;
				socklen_t client_addrlength = sizeof(client_address);
				ret = recvfrom(udpfd, buf, UDP_BUFFER_SIZE - 1, 0, 
						(struct sockaddr*)&client_address, &client_addrlength);
				if (ret > 0)
				{
					printf("[SERVER] echo UDP: %s\n", buf);
					sendto(udpfd, buf, UDP_BUFFER_SIZE - 1, 0,
						(struct sockaddr*)&client_address, client_addrlength);
				}
			}
			else if (events[i].events & EPOLLIN)
			{
				char buf[TCP_BUFFER_SIZE] = {'\0'};
				while (1)
				{
					memset(buf, '\0', TCP_BUFFER_SIZE);
					ret = recv(sockfd, buf, TCP_BUFFER_SIZE - 1, 0);
					if (ret < 0)
					{
						if (errno != EAGAIN && errno != EWOULDBLOCK)
						{
							close(sockfd);
						}
						break;
					}
					else if (ret == 0)
					{
						close(sockfd);
					}
					else
					{
						printf("[SERVER] echo TCP: %s\n", buf);
						send(sockfd, buf, ret, 0);
					}
				}
			}
			else
			{
				printf("[SERVER] something else happend\n");
			}
		}
	}
	
	close(listenfd);
	
	return 0;
}


















