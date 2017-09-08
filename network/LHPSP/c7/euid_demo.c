#include <unistd.h>
#include <stdio.h>
#include <fcntl.h>

int main()
{
	uid_t uid = getuid();
	uid_t euid = geteuid();
	printf("userid is %d, effective userid is: %d\n", uid, euid);

	int fd = open("./ROOTFILE", O_RDONLY);
	if (fd < 0)
	{
		perror("failed to open file");
	}
	else
	{
		printf("open file success!\n");
	}
	

	return 0;
}
