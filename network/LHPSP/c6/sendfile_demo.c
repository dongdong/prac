#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <libgen.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <stdbool.h>
#include <sys/sendfile.h>

int main(int argc, char* argv[])
{
	if (argc <= 3)
	{
		printf("usage: %s ip_address port_number filename\n", basename(argv[0]));
		return 1;
	}

	const char* ip = argv[1];
	int port = atoi(argv[2]);
	const char* file_name = argv[3];
	
	struct sockaddr_in address;
	bzero(&address, sizeof(address));
	address.sin_family = AF_INET;
	inet_pton(AF_INET, ip, &address.sin_addr);
	address.sin_port = htons(port);

	int sock = socket(AF_INET, SOCK_STREAM, 0);
	assert(sock >= 0);

	int ret = bind(sock, (struct sockaddr*)&address, sizeof(address));
	assert(ret == 0);

	ret = listen(sock, 5);
	assert(ret == 0);	

	struct sockaddr_in client;
	socklen_t client_addrlength = sizeof(client);
	int connfd = accept(sock, (struct sockaddr*)&client, &client_addrlength);
	if (connfd < 0)
	{
		perror("accept error: ");
		return 1;
	}

	int filefd = open(file_name, O_RDONLY);
	assert(filefd > 0);
	struct stat file_stat;
	fstat(filefd, &file_stat);	
	
	sendfile(connfd, filefd, NULL, file_stat.st_size);

	close(connfd);
	close(filefd);
	close(sock);	

	return 0;
}


