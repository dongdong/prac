#include <iostream>

void Demo1()
{
	int v1 = 0, v2 = 0;
	std::cout << "Enter two number:" << std::endl;
	std::cin >> v1 >> v2;
	std::cout << "The sum of " << v1 << " and " << v2 << " is "
		<< v1 + v2 << std::endl;
}

void Demo2()
{
	int value = 0, sum = 0;
	while (std::cin >> value)
	{
		sum += value;
	}
	std::cout << "sum is " << sum << std::endl;
}

int main()
{
	//Demo1();	
	Demo2();
	return 0;
}
