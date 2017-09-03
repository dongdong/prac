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

#define BUFFER_SIZE 1024

static const char* status_line[2] = {
	"200 OK",
	"500 Internal server error"
};

bool read_file(const char* file_name, char** p_file_buf)
{
	char* file_buf = NULL;
	struct stat file_stat;
	bool valid = false;
	if (stat(file_name, &file_stat) < 0)
	{
		return false;
	}
	else
	{
		if (S_ISDIR(file_stat.st_mode))
		{
			return false;
		}
		
		if (file_stat.st_mode & S_IROTH)
		{
			int fd = open(file_name, O_RDONLY);
			file_buf = malloc(file_stat.st_size + 1);
			if (file_buf)
			{
				memset(file_buf, '\0', file_stat.st_size + 1);
				if (read(fd, file_buf, file_stat.st_size) < 0)
				{
					free(file_buf);
					file_buf = NULL;
				}
				else
				{
					valid = true;
					*p_file_buf = file_buf;
				}
			}
			close(fd);
		}
	}

	return valid;
}

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
	
	char header_buf[BUFFER_SIZE] = {'\0'};
	char* file_buf = NULL;
	int len = 0;

	if (read_file(file_name, &file_buf))
	{

		//printf("read_file ok! contents: %s\n", file_buf);

		ret = snprintf(header_buf, BUFFER_SIZE - 1, "%s %s\r\n", 
				"HTTP/1.1", status_line[0]);
		len += ret;
		ret = snprintf(header_buf + len, BUFFER_SIZE - 1 - len, 
				"Content-Length: %d\r\n", (int)strlen(file_buf));
		len += ret;
		ret = snprintf(header_buf + len, BUFFER_SIZE - 1 - len, 
				"%s", "\r\n");
		
		//printf("set HTTP header ok!\n");

		struct iovec iv[2];
		iv[0].iov_base = header_buf;
		iv[0].iov_len = strlen(header_buf);
		iv[1].iov_base = file_buf;
		iv[1].iov_len = strlen(file_buf);		

		ret = writev(connfd, iv, 2);

		free(file_buf);
	}
	else
	{
		ret = snprintf(header_buf, BUFFER_SIZE - 1, "%s %s\r\n", 
				"HTTP/1.1", status_line[1]);
		len += ret;
		ret = snprintf(header_buf + len, BUFFER_SIZE - 1 - len, 
				"%s", "\r\n");
		send(connfd, header_buf, strlen(header_buf), 0);
	}
	
	close(connfd);	
	close(sock);	

	return 0;
}


