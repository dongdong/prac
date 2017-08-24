#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include <netinet/in.h>
#include <sys/socket.h>
#include <poll.h>
#include <unistd.h>
#include <sys/types.h>
#include <arpa/inet.h>

#define IP_ADDR		"127.0.0.1"
#define PORT		8787
#define MAX_LINE	1024
#define LIST_ENQ	5
#define OPEN_MAX	512
#define INF_TIM		-1

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

static void handle_client_conn(struct pollfd* conn_fds, int num)
{
	int i = 0;
	int n = 0;
	char buf[MAX_LINE] = {'\0'};

	for (i = 1; i <= num; i++)
	{
		if (conn_fds[i].fd < 0)
		{
			continue;
		}
		if (conn_fds[i].revents & POLLIN)
		{
			n = read(conn_fds[i].fd, buf, MAX_LINE);
			if (n == 0)
			{
				close(conn_fds[i].fd);
				conn_fds[i].fd = -1;
				continue;
			}
			printf("[SERVER] read msg: %s\n", buf);

			write(conn_fds[i].fd, buf, n);
		}
	}
}

static void handle_client_proc(int serv_fd)
{
	int conn_fd, sock_fd;
	struct sockaddr_in cli_addr;
	socklen_t cli_addr_len = sizeof(cli_addr);
	struct pollfd cli_fds[OPEN_MAX];
	int max_i = 0;
	int i = 0;
	int n_ready = 0;

	cli_fds[0].fd = serv_fd;
	cli_fds[0].events = POLLIN;

	for (i = 1; i < OPEN_MAX; i++)
	{
		cli_fds[i].fd = -1;
	}

	while (1)
	{
		n_ready = poll(cli_fds, max_i + 1, INF_TIM);
		if (n_ready == -1)
		{
			perror("[SERVER] poll error: ");
			return;
		}

		if (cli_fds[0].revents & POLLIN)
		{
			conn_fd	= accept(serv_fd, (struct sockaddr*)&cli_addr, &cli_addr_len);
			if (conn_fd == -1)
			{
				if (errno == EINTR)
				{
					continue;
				}
				else
				{
					perror("[SERVER] accept error: ");
					return;
				}
			}

			printf("[SERVER] accept a new client: %s:%d\n",
					inet_ntoa(cli_addr.sin_addr), cli_addr.sin_port);

			for (i = 1; i < OPEN_MAX; i++)
			{
				if (cli_fds[i].fd < 0)
				{
					cli_fds[i].fd = conn_fd;
					cli_fds[i].events = POLLIN;
					max_i = (i > max_i ? i : max_i);
					break;
				}
			}
			if (i == OPEN_MAX)
			{
				fprintf(stderr, "[SERVER] too many clients.\n");
			}
			
			if (n_ready == 1)
			{
				continue;
			}
		}

		handle_client_conn(cli_fds, max_i);
	}
}

int main(int argc, char** argv)
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













