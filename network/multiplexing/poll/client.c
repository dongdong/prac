#include <netinet/in.h>
#include <sys/socket.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>

#define MAX_LINE	1024
#define IP_ADDR		"127.0.0.1"
#define SERV_PORT	8787

int main(int argc,char *argv[])
{
	int sock_fd = 0;
	int ret_val = 0;
	struct sockaddr_in serv_addr;
	char recv_data[MAX_LINE] = {0};

	sock_fd = socket(AF_INET, SOCK_STREAM, 0);
	if (sock_fd == -1)
	{
		perror("[CLIENT] create socket error: ");
		return -1;
	}

	bzero(&serv_addr, sizeof(struct sockaddr_in));
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_port = htons(SERV_PORT);
	inet_pton(AF_INET, IP_ADDR, &serv_addr.sin_addr);
	
	ret_val = connect(sock_fd, (struct sockaddr*)&serv_addr, sizeof(struct sockaddr));
	if (ret_val == -1)
	{
		perror("[CLIENT] connect error: ");
		return -1;
	}

	while (1)
	{
		ret_val = send(sock_fd, "hello server from poll client", 64, 0);	
		printf("[CLIENT] send to server, %d\n", ret_val);
		
		memset(recv_data, 0, sizeof(recv_data));
		ret_val = recv(sock_fd, recv_data, MAX_LINE, 0);
		if (ret_val <= 0)
		{
			fprintf(stderr, "[CLIENT] server is closed!\n");
			break;
		}
		printf("[CLIENT] receive from server, %s\n", recv_data);

		sleep(5);
	}
			
	close(sock_fd);
	
	return 0;
}



