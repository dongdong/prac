#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <assert.h>
#include <stdio.h>
#include <signal.h>
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
static int pipefd[2];

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

void sig_handler(int sig)
{
	int save_errno = errno;
	int msg = sig;
	send(pipefd[1], (char*)&msg, 1, 0);
	errno = save_errno;
}

void add_sig(int sig)
{
	struct sigaction sa = {0};
	sa.sa_handler = sig_handler;
	sa.sa_flags |= SA_RESTART;
	sigfillset(&sa.sa_mask);
	int ret = sigaction(sig, &sa, NULL);
	assert(ret != -1);
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
	bzero(&address, sizeof(address));
	address.sin_family = AF_INET;
	inet_pton(AF_INET, ip, &address.sin_addr);
	address.sin_port = htons(port);

	int listenfd = socket(AF_INET, SOCK_STREAM, 0);
	assert(listenfd >= 0);
	ret = bind(listenfd, (struct sockaddr*)&address, sizeof(address));
	if (ret == -1)
	{
		perror("[SERVER] bind failed");
		return 1;
	}
	ret = listen(listenfd, 5);
	assert(ret != -1);

	struct epoll_event events[MAX_EVENT_NUMBER];
	int epollfd = epoll_create(5);
	assert(epollfd != -1);
	add_fd(epollfd, listenfd);

	ret = socketpair(PF_UNIX, SOCK_STREAM, 0, pipefd);
	assert(ret != -1);
	set_nonblocking(pipefd[1]);
	add_fd(epollfd, pipefd[0]);

	add_sig(SIGHUP);
	add_sig(SIGCHLD);
	add_sig(SIGTERM);
	add_sig(SIGINT);
	
	bool stop_server = false;
	while (!stop_server)
	{
		int number = epoll_wait(epollfd, events, MAX_EVENT_NUMBER, -1);
		if (number < 0 && errno != EINTR)
		{
			perror("[SERVER] epoll_wait failed");
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
			else if (sockfd == pipefd[0] && events[i].events & EPOLLIN)
			{
				int sig;
				char signals[1024] = {'\0'};
				ret = recv(pipefd[0], signals, sizeof(signals), 0);
				if (ret <= 0)
				{
					continue;
				}
				else
				{
					int j;
					for (j = 0; j < ret; ++j)
					{
						switch (signals[j])
						{
							case SIGCHLD:
								printf("[SERVER] SIGCHLD received, ignore\n");
								break;
							case SIGHUP:
								printf("[SERVER] SIGHUP received, ignore\n");
								break;
							case SIGTERM:
								printf("[SERVER] SIGTERM received, exit\n");
								stop_server = true;
								break;
							case SIGINT:
								printf("[SERVER] SIGINT received, exit\n");
								stop_server = true;
								break;
							default:
								printf("[SERVER] unknown signal receiverd\n");
								break;
						}
					}
				}
			}
			else
			{

			}
		} // end for
	} // end while

	printf("[SERVER] close fds\n");
	close(listenfd);
	close(pipefd[1]);
	close(pipefd[0]);	

	return 0;
}

