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
#include <sys/ioctl.h>
#include <libgen.h>
#include <stdbool.h>

#define BUFFER_SIZE 1024

int setnonblocking(int fd)
{
	int old_option = fcntl(fd, F_GETFL);
	int new_option = old_option | O_NONBLOCK;
	fcntl(fd, F_SETFL, new_option);
	return old_option;
}

int unblock_connect(const char* ip, int port, int time)
{
	int ret = 0;
	struct sockaddr_in address;
	bzero(&address, sizeof(address));
	address.sin_family = AF_INET;
	inet_pton(AF_INET, ip, &address.sin_addr);
	address.sin_port = htons(port);
	
	int sockfd = socket(AF_INET, SOCK_STREAM, 0);
	int fdopt = setnonblocking(sockfd);
	ret = connect(sockfd, (struct sockaddr*)&address, sizeof(address));

	if (ret == 0)
	{
		printf("[CLIENT] connect with server immediately\n");
		fcntl(sockfd, F_SETFL, fdopt);
		return sockfd;
	}
	else if (errno != EINPROGRESS)
	{
		perror("[CLIENT] connect failed");
		printf("[CLIENT] unblock connect not support\n");
		return -1;
	}
	else // ret == EINPROGRESS
	{
		fd_set writefds;
		struct timeval timeout;
		
		FD_ZERO(&writefds);
		FD_SET(sockfd, &writefds);

		timeout.tv_sec = time;
		timeout.tv_usec = 0;

		ret = select(sockfd + 1, NULL, &writefds, NULL, &timeout);
		if (ret < 0)
		{
			perror("[CLIENT] select failed");
			close(sockfd);
			return -1;
		}	
		if (!FD_ISSET(sockfd, &writefds))
		{
			printf("no events on sockfd found\n");
			close(sockfd);
			return -1;
		}

		int error = 0;
		socklen_t length = sizeof(error);
		if (getsockopt(sockfd, SOL_SOCKET, SO_ERROR, &error, &length) < 0)
		{
			perror("[CLIENT] get socket option failed");
			close(sockfd);
			return -1;
		}

		if (error != 0)
		{
			printf("[CLIENT] connection failed after select with the error: %d\n", 
					error);
			close(sockfd);
			return -1;	
		}

		printf("connection ready after select with the socket: %d\n", sockfd);
		fcntl(sockfd, F_SETFL, fdopt);		
		
		return sockfd;
	}	
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

	int sockfd = unblock_connect(ip, port, 10);
	if (sockfd < 0)
	{
		return -1;
	}

	printf("[CLIENT] connect success\n");

	close(sockfd);
	return 0;
}

