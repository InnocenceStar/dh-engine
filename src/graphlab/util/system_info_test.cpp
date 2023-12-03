#include "system_info.hpp"

//can g++ system_info_test.cpp -o system_info_test -std=c++11
// g++ system_info_test.cpp -o system_info_test -I/usr/local/cuda-11.1/targets/x86_64-linux/include -I~/dh-engine/src/ -std=c++11 -L/usr/local/cuda-11.1/targets/x86_64-linux/lib/stubs -lnvidia-ml
int main()
{
	graphlab::systeminfo::get_cpu_usage();
	graphlab::systeminfo::get_gpu_usage();
	return 0;
}