#ifndef GRAPHLAB_SYSTEM_INFO_HPP
#define GRAPHLAB_SYSTEM_INFO_HPP

#include <fstream>
#include <iostream>
#include <thread>
#include <chrono>
#include <string>

#include <unistd.h>
// /usr/local/cuda-11.1/targets/x86_64-linux/include/nvml.h
// /usr/include/hwloc/nvml.h
#ifdef _GPU
#include <nvml.h>
#endif
namespace graphlab
{
	namespace systeminfo
	{
		/**
		 * get cpu utilization rate
		 * @param null
		 * @return double: utilization rate
		 */
		float get_cpu_usage()
		{
			float usage = 0.0;
			// open /proc/stat file
			std::ifstream stat_file("/proc/stat");

			if (!stat_file.is_open())
			{
				std::cerr << "Failed to open /proc/stat file!" << std::endl;
				return usage;
			}

			std::string line;
			getline(stat_file, line);

			// parse CPU info
			if (line.substr(0, 3) == "cpu")
			{
				int user = 0, nice = 0, system = 0, idle = 0;
				sscanf(line.c_str(), "cpu %d %d %d %d", &user, &nice, &system, &idle);
				int total_idle = idle + nice;
				int total_usage = user + system;

				// wait for 0.1 sec，reg cpu info
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				stat_file.seekg(0);
				getline(stat_file, line);

				if (line.substr(0, 3) == "cpu")
				{
					int user2 = 0, nice2 = 0, system2 = 0, idle2 = 0;
					sscanf(line.c_str(), "cpu %d %d %d %d", &user2, &nice2, &system2, &idle2);

					int total_idle2 = idle2 + nice2;
					int total_usage2 = user2 + system2;

					// 计算 CPU 利用率
					usage = (float)(total_usage2 - total_usage) / (float)((total_usage2 + total_idle2) - (total_usage + total_idle)) * 100.0;
				}
			}

			// close /proc/stat file
			stat_file.close();

			std::cout<<"usage"<<usage<<std::endl;
			return usage;
		}
		/**
		 * @brief gpu usage
		 * @params null
		 * @return float: gpu usage
		*/
		float get_gpu_usage()
		{
			//TODO:if gpu is available
			///return get_gpu_usage_inner();
			return 0;
		}
		/*
		static float get_gpu_usage_inner()
		{
			std::cout<<"gpu hello"<<std::endl;
			nvmlReturn_t result;
			unsigned int device_count, i;
			// First initialize NVML library
			result = nvmlInit();

			result = nvmlDeviceGetCount(&device_count);
			if (NVML_SUCCESS != result)
			{
				std::cout << "Failed to query device count: " << nvmlErrorString(result);
			}
			std::cout << "Found" << device_count << " device" << std::endl;

			std::cout << "Listing devices:";
			//temporarily use the first gpu to return
			
			for (i = 0; i < device_count; i++)
			{
				nvmlDevice_t device;
				char name[NVML_DEVICE_NAME_BUFFER_SIZE];
				nvmlPciInfo_t pci;
				result = nvmlDeviceGetHandleByIndex(i, &device);
				if (NVML_SUCCESS != result)
				{
					std::cout << "get device failed " << std::endl;
				}
				result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
				if (NVML_SUCCESS == result)
				{
					std::cout << "GPU name： " << name << std::endl;
				}
				// usage
				nvmlUtilization_t utilization;
				result = nvmlDeviceGetUtilizationRates(device, &utilization);
				std::cout<<"result:"<<result<<std::endl;
				std::cout<<"success:"<<NVML_SUCCESS<<std::endl;
				if (NVML_SUCCESS == result)
				{
					std::cout << " 卡" << i << "使用率 ";
					std::cout << "GPU 使用率： " << utilization.gpu << "  显存使用率 " << utilization.memory << std::endl;
				}else{
					return 1;
				}
			}
			return utilization.gpu/100;
		}
		*/

	}

};
#endif