#include <iostream>


void sum1(const int a, const int b)
{
	std::cout << "sum1: " << a + b << std::endl;
}

void sum2(const int &a, const int &b)
{
	std::cout << "sum2: " << a + b << std::endl;
}

void DemoConstRef()
{
	int v1 = 5, v2 = 9;
	sum1(v1, v2);
	sum2(v1, v2);
}

int main()
{
	DemoConstRef();
	return 0;
}
