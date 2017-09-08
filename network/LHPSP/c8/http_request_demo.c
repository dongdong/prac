#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <libgen.h>

#define BUFFER_SIZE 4096

/*
enum CHECK_STATE 
{
	CHECK_STATE_REQUESTLINE = 0,
	CHECK_STATE_HEADER	
};

enum LINE_STATUS 
{
	LINE_OK = 0,
	LINE_BAD,
	LINE_OPEN
};

enum HTTP_CODE 
{
	NO_REQUEST,
	GET_REQUEST,
	BAD_REQUEST,
	FORBIDDEN_REQUEST,
	INTERNAL_ERROR,
	CLOSE_CONNECTION
};
*/

typedef int CHECK_STATE; 
#define	CHECK_STATE_REQUESTLINE 0
#define	CHECK_STATE_HEADER 1	

typedef int LINE_STATUS;
#define	LINE_OK 0
#define	LINE_BAD 1
#define	LINE_OPEN 2


typedef int HTTP_CODE;
#define	NO_REQUEST 0
#define	GET_REQUEST 1
#define	BAD_REQUEST 2
#define	FORBIDDEN_REQUEST 3
#define	INTERNAL_ERROR 4
#define	CLOSE_CONNECTION 5

static const char* szret[] = {
	"I get a correct result\n",
	"Something wrong"
};

LINE_STATUS parse_line(char* buffer, int* p_checked_index, int* p_read_index) 
{
	char temp;
	for (; (*p_checked_index) < (*p_read_index); ++(*p_checked_index)) 
	{
		temp = buffer[*p_checked_index];
		if (temp == '\r')
		{
			if ((*p_checked_index + 1) == *p_read_index)
			{
				return LINE_OPEN;
			}
			else if (buffer[*p_checked_index + 1] == '\n')
			{
				buffer[(*p_checked_index)++] = '\0';
				buffer[(*p_checked_index)++] = '\0';
				return LINE_OK;
			}
			else
			{
				return LINE_BAD;
			}
		}
		else if (temp == '\n')
		{
			if ((*p_checked_index > 1) && buffer[*p_checked_index - 1] == '\r')
			{
				buffer[(*p_checked_index)++] = '\0';
				buffer[(*p_checked_index)++] = '\0';
				return LINE_OK;
			}
			else
			{
				return LINE_BAD;
			}
		}
		else
		{
			// do nothing
		}
	} // end for
	
	return LINE_OPEN;
}

HTTP_CODE parse_requestline(char* temp, CHECK_STATE* p_check_state)
{
	char* url = strpbrk(temp, " \t");
	if (!url)
	{
		return BAD_REQUEST;
	}
	*url++ = '\0';

	char* method = temp;
	if (strcasecmp(method, "GET") == 0)
	{
		printf("The request method is GET\n");
	}
	else
	{
		return BAD_REQUEST;
	}
	url += strspn(url, " \t"); // trim

	char* version = strpbrk(url, " \t");
	if (!version)
	{
		return BAD_REQUEST;
	}
	*version++ = '\0';
	version += strspn(version, " \t");
	if (strcasecmp(version, "HTTP/1.1") != 0)
	{
		return BAD_REQUEST;
	}
	
	if (strncasecmp(url, "http://", 7) == 0)
	{
		url += 7;
		url = strchr(url, '/');
	}
	if (!url || url[0] != '/')
	{
		return BAD_REQUEST;
	}

	printf("The request URL is: %s\n", url);
	*p_check_state = CHECK_STATE_HEADER;
	
	return NO_REQUEST;
}


HTTP_CODE parse_headers(char* temp)
{
	if (temp[0] == '\0')
	{
		return GET_REQUEST;
	}
	else if (strncasecmp(temp, "Host:", 5) == 0)
	{
		temp += 5;
		temp += strspn(temp, " \t");
		printf("the request host is: %s\n", temp);
	}
	else
	{
		printf("ignore section: %s\n", temp);
	}

	return NO_REQUEST;
}

HTTP_CODE parse_content(char* buffer, int* p_checked_index, 
		CHECK_STATE* p_check_state, int* p_read_index, int* p_start_line)
{
	LINE_STATUS linestatus = LINE_OK;
	HTTP_CODE retcode = NO_REQUEST;

	while ((linestatus = parse_line(buffer, p_checked_index, p_read_index)) 
			== LINE_OK)
	{
		char* temp = buffer + *p_start_line;
		*p_start_line = *p_checked_index;
		
		switch (*p_check_state)
		{
			case CHECK_STATE_REQUESTLINE:
			{
				retcode = parse_requestline(temp, p_check_state);
				if (retcode == BAD_REQUEST)
				{
					return BAD_REQUEST;
				}
				break;
			}
			case CHECK_STATE_HEADER:
			{
				retcode = parse_headers(temp);
				if (retcode == BAD_REQUEST)
				{
					return BAD_REQUEST;
				}
				else if (retcode == GET_REQUEST)
				{
					return GET_REQUEST;
				}
				else
				{
					// do nothing
				}
				break;
			}
			default:
			{
				return INTERNAL_ERROR;
			}
		} // end switch
	} // end while

	if (linestatus = LINE_OPEN)
	{
		return NO_REQUEST;
	}
	else
	{
		return BAD_REQUEST;
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

	struct sockaddr_in address;
	bzero(&address, sizeof(address));
	address.sin_family = AF_INET;
	inet_pton(AF_INET, ip, &address.sin_addr);
	address.sin_port = htons(port);

	int listenfd = socket(AF_INET, SOCK_STREAM, 0);
	assert(listenfd >= 0);
	int ret = bind(listenfd, (struct sockaddr*)&address, sizeof(address));
	assert(ret != -1);
	ret = listen(listenfd, 5);
	assert(ret != -1);

	struct sockaddr_in client_address;
	socklen_t client_addrlength = sizeof(client_address);
	int fd = accept(listenfd, (struct sockaddr*)&client_address, 
			&client_addrlength);
	if (fd < 0)
	{
		perror("accept error:");
		close(listenfd);
		return 1;
	}
	
	char buffer[BUFFER_SIZE] = {'\0'};
	int data_read = 0;
	int read_index = 0;
	int checked_index = 0;
	int start_line = 0;
	CHECK_STATE checkstate = CHECK_STATE_REQUESTLINE;
	
	while (1)
	{
		data_read = recv(fd, buffer + read_index, BUFFER_SIZE - read_index, 0);
		if (data_read == -1)
		{
			printf("read failed!\n");
			break;
		}		
		else if (data_read == 0)
		{
			printf("remote client has closed the connection\n");
			break;
		}
		else
		{
			read_index += data_read;
			HTTP_CODE result = parse_content(buffer, &checked_index, &checkstate,
					&read_index, &start_line);
			
			if (result == NO_REQUEST)
			{
				continue;
			}
			else if (result == GET_REQUEST)
			{
				send(fd, szret[0], strlen(szret[0]), 0);
				break;
			}
			else
			{
				send(fd, szret[1], strlen(szret[1]), 0);
				break;
			}
		}
	}

	close(fd);
	close(listenfd);
	return 1;	
}


