#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <assert.h>

#define IP_ADDR		"127.0.0.1"
#define PORT		8787
#define MAX_LINE	1024
#define LIST_ENQ	5
#define SIZE		10


typedef struct server_context_st 
{
	int cli_cnt;
	int cli_fds[SIZE];
	fd_set all_fds;
	int max_fd;
} server_context_st;

static server_context_st* s_serv_ctx = NULL;

static int create_server_proc(const char* ip, int port)
{
	int fd;
	struct sockaddr_in serv_addr;
	int reuse = 1;

	fd = socket(AF_INET, SOCK_STREAM, 0);
	if (fd == -1) 
	{
		fprintf(stderr, "[SERVER] create socket fail, errno: %d, reason:%s\n",
				errno, strerror(errno));
		return -1;
	}

	if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) == -1)
	{
		fprintf(stderr, "[SERVER] setsocketopt fail, errno: %d, reason:%s\n",
				errno, strerror(errno));
		return -1;
	}

	bzero(&serv_addr, sizeof(serv_addr));
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_port = htons(port);
	inet_pton(AF_INET, ip, &serv_addr.sin_addr);
	
	if (bind(fd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) ==  -1)
	{
		perror("[SERVER] bind error: ");
		return -1;
	}

	listen(fd, LIST_ENQ);

	return fd;
}


static int accept_client_proc(int serv_fd)
{
	struct sockaddr_in cli_addr;
	socklen_t cli_addr_len = sizeof(cli_addr);
	int cli_fd = -1;
	int i = 0;

	while (1)
	{
		cli_fd = accept(serv_fd, (struct sockaddr*)&cli_addr, &cli_addr_len);

		if (cli_fd == -1) 
		{
			if (errno == EINTR)
			{
				continue;
			}
			else
			{
				fprintf(stderr, "[SERVER] accept fail, errno: %d, reason:%s\n",
						errno, strerror(errno));
				return -1;
			}
		}
			
		break;
	}
		
	printf("[SERVER] accept a new client: %s:%d\n",
			inet_ntoa(cli_addr.sin_addr), cli_addr.sin_port);

	for (i = 0; i < SIZE; i++)
	{
		if (s_serv_ctx->cli_fds[i] < 0)
		{
			s_serv_ctx->cli_fds[i] = cli_fd;
			s_serv_ctx->cli_cnt++;
			break;
		}
	}
	
	if (i == SIZE)
	{
		fprintf(stderr, "[SERVER] too many clients!\n");
		return -1;
	}

	return 0;
}

static int handle_client_msg(int fd, char* buf)
{
	assert(buf);
	printf("[SERVER] recv buf is %s\n", buf);
	write(fd, buf, strlen(buf) + 1);
	return 0;
}

static void recv_client_msg(fd_set* read_fds)
{
	int i = 0, n =0;
	int cli_fd;
	char buf[MAX_LINE] = {0};

	for (i = 0; i <= s_serv_ctx->cli_cnt; i++) 
	{
		cli_fd = s_serv_ctx->cli_fds[i];
		if (cli_fd <= 0)
		{
			continue;
		}
		if (FD_ISSET(cli_fd, read_fds))
		{
			n = read(cli_fd, buf, MAX_LINE);
			if (n <= 0)
			{
				FD_CLR(cli_fd, &s_serv_ctx->all_fds);
				close(cli_fd);
				s_serv_ctx->cli_fds[i] = -1;
				continue;
			}
			handle_client_msg(cli_fd, buf);
		}
	}
}

static void handle_client_proc(int serv_fd)
{
	int cli_fd = -1;
	int ret_val = 0;
	fd_set* read_fds = &s_serv_ctx->all_fds;
	struct timeval tv;
	int i = 0;

	while (1)
	{
		FD_ZERO(read_fds);
		FD_SET(serv_fd, read_fds);
		s_serv_ctx->max_fd = serv_fd;

		tv.tv_sec = 30;
		tv.tv_usec = 0;

		for (i = 0; i < s_serv_ctx->cli_cnt; i++)
		{
			cli_fd = s_serv_ctx->cli_fds[i];
			if (cli_fd != -1)
			{
				FD_SET(cli_fd, read_fds);
			}
			if (s_serv_ctx->max_fd < cli_fd)
			{
				s_serv_ctx->max_fd = cli_fd;
			}
		}

		ret_val = select(s_serv_ctx->max_fd + 1, read_fds, NULL, NULL, &tv);
		if (ret_val == -1)
		{
			fprintf(stderr, "[SERVER] select error, errno: %d, reason:%s\n",
					errno, strerror(errno));
			return;
		}
		if (ret_val == 0)
		{
			printf("[SERVER] select timeout\n");
			continue;
		}

		if (FD_ISSET(serv_fd, read_fds))
		{
			accept_client_proc(serv_fd);
		} 
		else 
		{
			recv_client_msg(read_fds);
		}
	}
}

static int server_init()
{
	int i = 0;

	s_serv_ctx = (server_context_st *)malloc(sizeof(server_context_st));
	if (s_serv_ctx == NULL)
	{
		fprintf(stderr, "[SERVER] init error! malloc failed.\n");
		return -1;
	}
	memset(s_serv_ctx, 0, sizeof(server_context_st));

	for (i = 0; i < SIZE; i++)
	{
		s_serv_ctx->cli_fds[i] = -1;
	}

	return 0;
}

static void server_destroy()
{
	if (s_serv_ctx) 
	{
		free(s_serv_ctx);
		s_serv_ctx = NULL;
	}
}

int main(int argc, char** argv)
{
	int serv_fd;

	if (server_init() < 0)
	{
		return -1;
	}

	serv_fd = create_server_proc(IP_ADDR, PORT);
	if (serv_fd < 0)
	{
		server_destroy();
		return -1;
	}

	printf("[SERVER] server has establised, waiting for messages from client.\n");

	handle_client_proc(serv_fd);
	server_destroy();

	return 0;
}

