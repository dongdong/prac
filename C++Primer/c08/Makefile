CPPFLAG=-std=c++11

all: Sales_data Sales_data_file Person_info

Sales_data: Sales_data_test.cc Sales_data.h Sales_data.cc
	g++ $(CPPFLAG) -o Sales_data Sales_data_test.cc Sales_data.cc

Sales_data_file: Sales_data_file.cc Sales_data.h Sales_data.cc
	g++ $(CPPFLAG) -o Sales_data_file Sales_data_file.cc Sales_data.cc

Person_info: Person_info.cc
	g++ $(CPPFLAG) -o Person_info Person_info.cc

clean:
	rm -rf Sales_data Sales_data_file Person_info
