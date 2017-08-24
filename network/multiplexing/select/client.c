#include <netinet/in.h>
#include <sys/socket.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/select.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <errno.h>

#define MAX_LINE 	1024
#define IP_ADDR		"127.0.0.1"
#define SERV_PORT	8787

#define max(a, b) ((a) > (b) ? (a) : (b))

static void handle_recv_msg(int sock_fd, char* buf)
{
	printf("[CLIENT] client recv msg is: %s\n", buf);
	sleep(5);
	write(sock_fd, buf, strlen(buf) + 1);
}

static void handle_connection(int sock_fd)
{
	char recv_line[MAX_LINE] = {'\0'};
	int max_fdp;
	fd_set read_fds;
	struct timeval tv;
	int ret_val = 0;
	int n = 0;

	while (1) 
	{
		FD_ZERO(&read_fds);
		FD_SET(sock_fd, &read_fds);
		max_fdp = sock_fd;

		tv.tv_sec = 5;
		tv.tv_usec = 0;

		ret_val = select(max_fdp + 1, &read_fds, NULL, NULL, &tv);

		if (ret_val == -1)
		{
			return;
		}
		if (ret_val == 0)
		{
			printf("[CLIENT] client timeout.\n");
			continue;
		}
		if (FD_ISSET(sock_fd, &read_fds))
		{
			n = read(sock_fd, recv_line, MAX_LINE);
			if (n <= 0)
			{
				fprintf(stderr, "[CLIENT] server is closed.\n");
				close(sock_fd);
				FD_CLR(sock_fd, &read_fds);
				return;
			}
			handle_recv_msg(sock_fd, recv_line);
		}
	}
}

int main(int argc, char** argv)
{
	int sock_fd;
	struct sockaddr_in serv_addr;
	int ret_val = 0;

	sock_fd = socket(AF_INET, SOCK_STREAM, 0);
	
	bzero(&serv_addr, sizeof(serv_addr));
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_port = htons(SERV_PORT);
	inet_pton(AF_INET, IP_ADDR, &serv_addr.sin_addr);

	ret_val = connect(sock_fd, (struct sockaddr*)&serv_addr, sizeof(serv_addr));
	if (ret_val < 0)
	{
		fprintf(stderr, "[CLIENT] connect fail, error:%s\n", strerror(errno));
		return -1;
	}

	printf("[CLIENT] client send to server.\n");
	write(sock_fd, "hello server", 32);

	handle_connection(sock_fd);
	return 0;
}








