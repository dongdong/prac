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
#include <stdbool.h>
#include <signal.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/mman.h>

#define BUFFER_SIZE 1024
#define USER_LIMIT 5
#define FD_LIMIT 65535
#define MAX_EVENT_NUMBER 1024
#define PROCESS_LIMIT 65535

struct client_data
{
	struct sockaddr_in address;
	int connfd;
	pid_t pid;
	int pipefd[2];
};

static const char* shm_name="/my_shm";
int sig_pipefd[2];
int epollfd;
int listenfd;
int shmfd;
char* share_mem = NULL;
struct client_data* users = NULL;
int* sub_process = NULL;
int user_count = 0;
bool stop_child = false;

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
	send(sig_pipefd[1], (char*)&msg, 1, 0);
	errno = save_errno;
}

void add_sig(int sig, void(*handler)(int), bool restart)
{
	struct sigaction sa = {0};
	sa.sa_handler = handler;
	if (restart) {
		sa.sa_flags |= SA_RESTART;
	}
	sigfillset(&sa.sa_mask);
	int ret = sigaction(sig, &sa, NULL);
	assert(ret != -1);
}

void del_resource()
{
	close(sig_pipefd[0]);
	close(sig_pipefd[1]);
	close(listenfd);
	close(epollfd);
	shm_unlink(shm_name);
	free(users);
	free(sub_process);
}

void child_term_handler(int sig)
{
	stop_child = true;
}

int run_child(int idx, struct client_data* users, char* share_mem)
{
	struct epoll_event events[MAX_EVENT_NUMBER];
	int child_epollfd = epoll_create(5);
	assert(child_epollfd != -1);
	int connfd = users[idx].connfd;
	add_fd(child_epollfd, connfd);
	int pipefd = users[idx].pipefd[1];
	add_fd(child_epollfd, pipefd);

	int ret;
	add_sig(SIGTERM, child_term_handler, false);
	
	while (!stop_child)
	{
		int number = epoll_wait(child_epollfd, events, MAX_EVENT_NUMBER, -1);
		if (number < 0 && errno != EINTR)
		{
			perror("[SERVER][CHILD] epoll_wait failed");
			break;
		}
		int i;
		for (i = 0; i < number; ++i)
		{
			int sockfd = events[i].data.fd;
			if ((sockfd == connfd) && (events[i].events & EPOLLIN))
			{
				memset(share_mem + idx*BUFFER_SIZE, '\0', BUFFER_SIZE);
				ret = recv(connfd, share_mem + idx * BUFFER_SIZE, 
						BUFFER_SIZE - 1, 0);
				if (ret < 0)
				{
					if (errno != EAGAIN)
					{
						perror("[SERVER][CHILD] recv failed");
						stop_child = true;
					}
				}
				else if (ret == 0)
				{
					printf("[SERVER][CHILD] connection close by client, quit\n");
					stop_child = true;
				}
				else {
					send(pipefd, (char*)&idx, sizeof(idx), 0);
				}
			}
			else if ((sockfd == pipefd) && (events[i].events & EPOLLIN))
			{
				int client = 0;
				ret = recv(sockfd, (char*)&client, sizeof(client), 0);
				if (ret < 0)
				{
					if (errno != EAGAIN)
					{
						perror("[SERVER][CHILD] recv from pipe failed");
						stop_child = true;
					}
				}
				else if (ret == 0)
				{
					printf("[SERVER][CHILD] pipe closed by main, quit\n");
					stop_child = true;
				}
				else {
					send(connfd, share_mem + client * BUFFER_SIZE, BUFFER_SIZE, 0);
				}
			}
			else
			{

			}
		} // end for
	} // end while

	close(connfd);
	close(pipefd);
	close(child_epollfd);

	return 0;
}


int main(int argc, char** argv)
{
	if (argc <= 2)
	{
		printf("usage: %s ip port number\n", basename(argv[0]));
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

	listenfd = socket(AF_INET, SOCK_STREAM, 0);
	assert(listenfd > 0);
	ret = bind(listenfd, (struct sockaddr*)&address, sizeof(address));
	assert(ret != -1);
	ret =listen(listenfd, 5);
	assert(ret != -1);

	user_count = 0;
	users = (struct client_data*)malloc(sizeof(struct client_data) * (USER_LIMIT + 1));
	sub_process = (int*)malloc(sizeof(int) * PROCESS_LIMIT);
	int i;
	for (i = 0; i < PROCESS_LIMIT; ++i)
	{
		sub_process[i] = -1;
	}
	
	struct epoll_event events[MAX_EVENT_NUMBER];
	epollfd = epoll_create(5);
	assert(epollfd != -1);
	add_fd(epollfd, listenfd);
	
	ret = socketpair(PF_UNIX, SOCK_STREAM, 0, sig_pipefd);
	assert(ret != -1);
	set_nonblocking(sig_pipefd[1]);
	add_fd(epollfd, sig_pipefd[0]);

	add_sig(SIGCHLD, sig_handler, true);
	add_sig(SIGTERM, sig_handler, true);
	add_sig(SIGINT, sig_handler, true);
	add_sig(SIGPIPE, SIG_IGN, true);

	bool stop_server = false;
	bool terminate = false;
	
	shmfd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
	assert(shmfd != -1);
	ret = ftruncate(shmfd, USER_LIMIT * BUFFER_SIZE);
	assert(ret != -1);

	share_mem = (char*)mmap(NULL, USER_LIMIT * BUFFER_SIZE, PROT_READ | PROT_WRITE, 
			MAP_SHARED, shmfd, 0);
	assert(share_mem != MAP_FAILED);
	close(shmfd);

	while (!stop_server)
	{
		int number = epoll_wait(epollfd, events, MAX_EVENT_NUMBER, -1);
		if (number < 0 && errno != EINTR) {
			perror("[SERVER][MAIN] epoll_wait failed");
			break;
		}
		int i;
		for (i = 0; i < number; ++i) 
		{
			int sockfd = events[i].data.fd;
			if (sockfd == listenfd) 
			{
				struct sockaddr_in client_address;
				socklen_t client_addrlength = sizeof(client_address);
				int connfd = accept(listenfd, (struct sockaddr*)&client_address,
						&client_addrlength);
				if (connfd < 0) 
				{
					perror("[SERVER][MAIN] accept failed");
					continue;
				}
				if (user_count >= USER_LIMIT) 
				{
					const char* info = "too many users\n";
					printf("[SERVER][MAIN] %s", info);
					send(connfd, info, strlen(info), 0);
					close(connfd);
					continue;
				}
				users[user_count].address = client_address;
				users[user_count].connfd = connfd;
				ret = socketpair(PF_UNIX, SOCK_STREAM, 0, users[user_count].pipefd);
				assert(ret != -1);
				pid_t pid = fork();
				if (pid < 0) 
				{
					perror("[SERVER][MAIN] fork failed");
					close(connfd);
					continue;
				}
				else if (pid == 0)
				{
					printf("[SERVER][CHILD] fork success, run child\n");
					close(epollfd);
					close(listenfd);
					close(users[user_count].pipefd[0]);
					close(sig_pipefd[0]);
					close(sig_pipefd[1]);
					run_child(user_count, users, share_mem);
					munmap((void*)share_mem, USER_LIMIT * BUFFER_SIZE);
					exit(0);
				}
				else 
				{
					close(connfd);
					close(users[user_count].pipefd[1]);
					add_fd(epollfd, users[user_count].pipefd[0]);
					users[user_count].pid = pid;
					sub_process[pid] = user_count;
					++user_count;
				}
			}
			else if ((sockfd == sig_pipefd[0]) && (events[i].events & EPOLLIN))
			{
				int sig;
				char signals[1024] = {'\0'};
				ret = recv(sig_pipefd[0], signals, sizeof(signals), 0);
				if (ret == -1)
				{
					continue;
				}
				else if (ret == 0)
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
							{
								pid_t pid;
								int stat;
								while ((pid = waitpid(-1, &stat, WNOHANG)) > 0)
								{
									int del_user = sub_process[pid];
									sub_process[pid] = -1;
									if (del_user < 0 || del_user > USER_LIMIT)
									{
										continue;
									}
									epoll_ctl(epollfd, EPOLL_CTL_DEL, 
											users[del_user].pipefd[0], 0);
									close(users[del_user].pipefd[0]);
									users[del_user] = users[--user_count];
									sub_process[users[del_user].pid] = del_user;
								}
								if (terminate && user_count == 0)
								{
									printf("[SERVER][MAIN] no child now, exit\n");
									stop_server = true;
								}
								break;
							}
							case SIGTERM:
							case SIGINT:
							{
								if (user_count == 0)
								{
									printf("[SERVER][MAIN] no child, just exit\n");
									stop_server = true;
									break;
								}
								printf("[SERVER][MAIN] kill all child now\n");
								int k;
								for (k = 0; k < user_count; ++k)
								{
									int pid = users[k].pid;
									kill(pid, SIGTERM);
								}
								terminate = true;
								break;
							}
							default:
							{
								break;
							}
						} // end switch
					} // end for
				} // end else
			}
			else if (events[i].events & EPOLLIN)
			{
				int child = 0;
				ret = recv(sockfd, (char*)&child, sizeof(child), 0);
				printf("[SERVER][MAIN] read data from child across pipe\n");
				if (ret == -1)
				{
					continue;
				}
				if (ret == 0)
				{
					continue;
				}
				else
				{
					int x;
					for (x = 0; x < user_count; ++x)
					{
						if (users[x].pipefd[0] != sockfd)
						{
							printf("[SERVER][MAIN] send data to child across pipe\n");
							send(users[x].pipefd[0], (char*)&child, sizeof(child), 0);
						}
					}
				}
			}
			else
			{

			}
		} // end for
	} // end while

	del_resource();
	return 0;
}


