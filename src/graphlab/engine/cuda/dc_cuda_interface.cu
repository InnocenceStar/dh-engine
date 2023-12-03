/**
 * Copyright (c) 2022 Linyi University.
 *     All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an "AS
 *  IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *  express or implied.  See the License for the specific language
 *  governing permissions and limitations under the License.
 */
#include "dc_cuda_interface.cuh"
#ifdef GPU
#ifndef DC_CUDA_INTERFACE_CU
#define DC_CUDA_INTERFACE_CU
//#include <graphlab/macros_def.hpp>
#include <boost/foreach.hpp>

/**
 * \brief redefine new for getting the size of new.
 * \param [in] size size of new 
 * \param [in,out] file file name
 * \param [in] line the line of new code location
*/
void * operator new(size_t size, const char* file, int line)
{
    void *p = operator new(size);
    printf("new size: %d\t file : %s \t line : %d\t new addr:%p \t funcname:%s \n",size,file,line,p,__func__);
    return p;
}
/**
 * \brief redefine delete.
 * \param [in] p addr
*/
void operator delete(void* p)
{
    printf("delete: %p \t func_name:%s\n",p, __func__ );
    free(p);
}
/**
 * \brief redefine delete.
 * \param [in] p addr
*/
void operator delete[](void* p) 
{
    printf("delete [] : %p\t func_name:%s\n",p, __func__ );
    free(p);
}
/**
 * \brief redefine delete.
 * \param [in] p addr
 * \param [in] file addr
 * \param [in] line line of delete
*/
void operator delete(void* p, const char* file, int line)
{
    printf("delete : %p\t func_name:%s\n",p, __func__ );
    free(p);
}
/**
 * \brief redefine delete.
 * \param [in] p addr
 * \param [in] file file name
 * \param [in] line line of delete
*/
void operator delete [](void* p, const char* file, int line)
{
    printf("delete: %p\t func_name:%s\n",p, __func__ );
    free(p);
}
#define new new(__FILE__, __LINE__)


int cuda_env_init()
{
	// get the nums of gpu
	int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
	if(error_id!=cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n ->%s\n",
              (int)error_id,cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }
    if(deviceCount==0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n",deviceCount);
    }
	int dev=0,driverVersion=0,runtimeVersion=0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("Device %d:\"%s\"\n",dev,deviceProp.name);
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    // printf("CUDA Driver Version / Runtime Version\t %d.%d  /  %d.%d\n",
    //     driverVersion/1000,(driverVersion%100)/10,
    //     runtimeVersion/1000,(runtimeVersion%100)/10);
    // printf("CUDA Capability Major/Minor version number:\t %d.%d\n",
    //     deviceProp.major,deviceProp.minor);
    // printf("Total amount of global memory:\t %.2f MBytes (%llu bytes)\n",
    //         (float)deviceProp.totalGlobalMem/pow(1024.0,3));
    // printf("GPU Clock rate:\t %.0f MHz (%0.2f GHz)\n",
    //         deviceProp.clockRate*1e-3f,deviceProp.clockRate*1e-6f);
    // printf("Memory Bus width:\t %d-bits\n", deviceProp.memoryBusWidth);
    if (deviceProp.l2CacheSize)
    {
        printf("L2 Cache Size:\t %d bytes\n", deviceProp.l2CacheSize);
    }
	return deviceCount;
}

typedef graphlab::distributed_graph<float, float> graph_float_type;

//template void cuda_data_init<std::vector<size_t>*,graph_float_type,struct cuda_graph_data_type<float,float,float>>(std::vector<size_t>*,graph_float_type,struct cuda_graph_data_type<float,float,float>);  

/**
 * \brief Explicitly specify cuda data initialization
 * \param vertex_index_list Graph vertex index list executed by GPU
 * \param graph local graph
 * \param cuda_graph_data Graph data stored in GPU 
*/
template<> void cuda_data_init<std::vector<unsigned long, std::allocator<unsigned long> >*, graphlab::distributed_graph<float, float>, cuda_graph_data_type<float, float, float> >
(std::vector<unsigned long, std::allocator<unsigned long> >* vertex_index_list, graphlab::distributed_graph<float, float>& graph, cuda_graph_data_type<float, float, float>* cuda_graph_data)
{
    typedef typename graphlab::distributed_graph<float, float>::vertex_id_type vertex_id_type;
    typedef typename graphlab::distributed_graph<float, float>::vertex_data_type    vertex_data_type;
    typedef typename graphlab::distributed_graph<float, float>::edge_data_type    edge_data_type;
    typedef typename graphlab::distributed_graph<float, float>::local_vertex_type    local_vertex_type;
    typedef typename graphlab::distributed_graph<float, float>::local_edge_type    local_edge_type;
    typedef typename graphlab::distributed_graph<float, float>::vertex_type          vertex_type;
    typedef typename graphlab::distributed_graph<float, float>::edge_type            edge_type;
    //typedef typename Graph_Type::            edge_type;
    typedef std::vector<unsigned long, std::allocator<unsigned long> > Cuda_Vertex;

    ////-------------------------------cpu data init----------------------------------
    
    std::cout<<"cuda sp template data init"<<std::endl;
    int vertex_list_len = vertex_index_list->size();
    
    std::vector<vertex_data_type>* vertex_data_list = new std::vector<vertex_data_type>(vertex_list_len,-1);

    std::vector<size_t>* edge_id_list = new std::vector<size_t>();
    std::vector<edge_data_type>* edge_data_list = new std::vector<edge_data_type>(vertex_list_len,-1);

    Cuda_Vertex* adj_vertex_id_list = new Cuda_Vertex(vertex_list_len,-1);
    std::vector<vertex_data_type>* adj_data_list = new std::vector<vertex_data_type>(vertex_list_len,-1);
    
    cuda_graph_data->vertex_id_list_incpu = vertex_index_list;
    cuda_graph_data->vertex_data_list_incpu = vertex_data_list;
    cuda_graph_data->edge_id_list_incpu = edge_id_list;
    cuda_graph_data->edge_data_list_incpu = edge_data_list;
    cuda_graph_data->adj_vertex_id_list_incpu = adj_vertex_id_list;
    cuda_graph_data->adj_vertex_data_list_incpu = adj_data_list;

    int index = 0;
    //edge index: 0 5
    //adj vertex_index_list index: 0 1 2 3 4 | 5
    // int  index :     1 2 3 4 5 |
    for(auto vertex_id = vertex_index_list->begin(); vertex_id != vertex_index_list->end();++vertex_id)
    {
        edge_id_list->push_back(index);

        local_vertex_type local_vertex = graph.l_vertex(*vertex_id);

        vertex_data_list->push_back(local_vertex.data());

        BOOST_FOREACH(local_edge_type local_edge, local_vertex.in_edges()) {
            ++index;
            edge_type edge(local_edge);
            edge_data_list->push_back(edge.data());
            adj_vertex_id_list->push_back(edge.source().id());
            adj_data_list->push_back(edge.source().data());


        }


        // foreach(local_edge_type local_edge, local_vertex.out_edges()) {
        //     ++index;
        //     edge_type edge(local_edge);
        //     edge_data_list.push_back(edge.data());
        //     adj_vertex_id_list.push_back(edge.target().id());
        //     adj_data_list.push_back(edge.source().data());
        // }
    }

    if(cuda_graph_data->gpu_status == GPU_EXCUTE_APPLY_READY){
        std::vector<vertex_data_type>* gather_num = new std::vector<vertex_data_type>(vertex_list_len,0);
        int i = 0;
        for(auto vertex_id =  vertex_index_list->begin();vertex_id != vertex_index_list->end();++vertex_id)
        {
          vertex_data_type vdt =  (*(cuda_graph_data->apply_gather_nums_incpu))[i]; 
          gather_num->push_back(vdt);
          ++i;
        }
        cuda_graph_data->apply_gather_nums_incpu = gather_num;
    }

    ///-------------------------------gpu data init------------------------------------
	// init data
    // struct cuda_graph_data_type<vertex_data_type> *cuda_graph_data = NULL;

    cuda_graph_data->vertex_list_len = vertex_index_list->size();

    cuda_graph_data->edge_index_len = edge_id_list->size();

    cuda_graph_data->edge_data_len = edge_data_list->size();

    cuda_graph_data->adj_vertex_list_len = adj_vertex_id_list->size();

    vertex_id_type* cuda_vertex_id_list = NULL;
    vertex_data_type* cuda_vertex_data_list = NULL;

    size_t* cuda_edge_id_list = NULL;
    edge_data_type* cuda_edge_data_list = NULL;
    
    vertex_id_type* cuda_adj_vertex_id_list = NULL;
    vertex_data_type* cuda_adj_data_list = NULL;

    cudaMalloc((void**)&cuda_vertex_id_list,  get_vertex_id_list_byte_len(cuda_graph_data));
    cudaMalloc((void**)&cuda_vertex_data_list,  get_vertex_data_list_byte_len(cuda_graph_data));
    cudaMalloc((void**)&cuda_edge_id_list,  get_edge_id_list_byte_len(cuda_graph_data));
    cudaMalloc((void**)&cuda_edge_data_list,  get_edge_data_list_byte_len(cuda_graph_data));
    cudaMalloc((void**)&cuda_adj_vertex_id_list,  get_adj_vertex_id_list_byte_len(cuda_graph_data));
    cudaMalloc((void**)&cuda_adj_data_list,  get_adj_vertex_data_list_byte_len(cuda_graph_data));
    // ret gather
    cudaMalloc((void**)&cuda_graph_data->gather_nums_ingpu,  get_gather_num_byte_len(cuda_graph_data));



    cudaMemcpy(cuda_vertex_id_list,vertex_index_list->data(), get_vertex_id_list_byte_len(cuda_graph_data),cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_vertex_data_list,vertex_data_list->data(), get_vertex_data_list_byte_len(cuda_graph_data),cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_edge_id_list,edge_id_list->data(), get_edge_id_list_byte_len(cuda_graph_data),cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_edge_data_list,edge_data_list->data(), get_edge_data_list_byte_len(cuda_graph_data),cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_adj_vertex_id_list,adj_vertex_id_list->data(), get_adj_vertex_id_list_byte_len(cuda_graph_data),cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_adj_data_list,adj_data_list->data(), get_adj_vertex_data_list_byte_len(cuda_graph_data),cudaMemcpyHostToDevice);

    if(cuda_graph_data->gpu_status == GPU_EXCUTE_APPLY_READY){
        cudaMemcpy(cuda_graph_data->gather_nums_ingpu,(cuda_graph_data->apply_gather_nums_incpu)->data(),
                     get_gather_num_byte_len(cuda_graph_data),cudaMemcpyHostToDevice);

    }

    cuda_graph_data->vertex_id_list_ingpu = cuda_vertex_id_list;
    cuda_graph_data->vertex_data_list_ingpu = cuda_vertex_data_list;
    cuda_graph_data->edge_id_list_ingpu = cuda_edge_id_list;
    cuda_graph_data->edge_data_list_ingpu = cuda_edge_data_list;
    cuda_graph_data->adj_vertex_id_list_ingpu = cuda_adj_vertex_id_list;
    cuda_graph_data->adj_vertex_data_list_ingpu = cuda_adj_data_list;

}
/**
 * \brief cuda data initialization
 * \param vertex_index_list Graph vertex index list executed by GPU
 * \param graph local graph
 * \param cuda_graph_data Graph data stored in GPU 
*/
template<typename Cuda_Vertex,typename Graph_Type,typename vertex_graph_type>
void cuda_data_init(Cuda_Vertex vertex_index_list, Graph_Type& graph,
                    vertex_graph_type *cuda_graph_data)
{
    typedef typename Graph_Type::vertex_id_type vertex_id_type;
    typedef typename Graph_Type::vertex_data_type    vertex_data_type;
    typedef typename Graph_Type::edge_data_type    edge_data_type;
    typedef typename Graph_Type::local_vertex_type    local_vertex_type;
    typedef typename Graph_Type::local_edge_type    local_edge_type;
    typedef typename Graph_Type::vertex_type          vertex_type;
    typedef typename Graph_Type::edge_type            edge_type;
    //typedef typename Graph_Type::            edge_type;


    ////-------------------------------cpu data init----------------------------------
    
    int vertex_list_len = vertex_index_list->size();
    std::cout<<"cuda_data_init----------------------------------------"<<std::endl;
    std::cout<<"vertex_list_len : "<<vertex_list_len<<std::endl;
    
    std::vector<vertex_data_type>* vertex_data_list = new std::vector<vertex_data_type>(vertex_list_len,-1);

    std::vector<size_t>* edge_id_list = new std::vector<size_t>();
    std::vector<edge_data_type>* edge_data_list = new std::vector<edge_data_type>(vertex_list_len,-1);

    Cuda_Vertex* adj_vertex_id_list = new Cuda_Vertex(vertex_list_len,-1);
    std::vector<vertex_data_type>* adj_data_list = new std::vector<vertex_data_type>(vertex_list_len,-1);
    
    cuda_graph_data->vertex_id_list_incpu = vertex_index_list;
    cuda_graph_data->vertex_data_list_incpu = vertex_data_list;
    cuda_graph_data->edge_id_list_incpu = edge_id_list;
    cuda_graph_data->edge_data_list_incpu = edge_data_list;
    cuda_graph_data->adj_vertex_id_list_incpu = adj_vertex_id_list;
    cuda_graph_data->adj_vertex_data_list_incpu = adj_data_list;

    int index = 0;
    //edge index: 0 5
    //adj vertex_index_list index: 0 1 2 3 4 | 5
    // int  index :     1 2 3 4 5 |
    for(auto vertex_id = vertex_index_list->begin(); vertex_id != vertex_index_list->end();++vertex_id)
    {
        edge_id_list->push_back(index);

        local_vertex_type local_vertex = graph.l_vertex(*vertex_id);

        vertex_data_list->push_back(local_vertex.data());

        BOOST_FOREACH(local_edge_type local_edge, local_vertex.in_edges()) {
            ++index;
            edge_type edge(local_edge);
            edge_data_list->push_back(edge.data());
            adj_vertex_id_list->push_back(edge.source().id());
            adj_data_list->push_back(edge.source().data());


        }


        // foreach(local_edge_type local_edge, local_vertex.out_edges()) {
        //     ++index;
        //     edge_type edge(local_edge);
        //     edge_data_list.push_back(edge.data());
        //     adj_vertex_id_list.push_back(edge.target().id());
        //     adj_data_list.push_back(edge.source().data());
        // }
    }

    if(cuda_graph_data->GPU_EXCUTE_APPLY_READY){
        std::vector<vertex_data_type>* gather_num = new std::vector<vertex_data_type>(vertex_list_len,0);
        for(auto vertex_id =  vertex_index_list->begin();vertex_id != vertex_index_list->end();++vertex_id)
        {
          gather_num->push_back((cuda_graph_data->apply_gather_nums_incpu)[vertex_id]);
        }
        cuda_graph_data->apply_gather_nums_incpu = gather_num->data();
    }

    ///-------------------------------gpu data init------------------------------------
	// init data
    // struct cuda_graph_data_type<vertex_data_type> *cuda_graph_data = NULL;

    cuda_graph_data->vertex_list_len = vertex_index_list->size();

    cuda_graph_data->edge_index_len = edge_id_list->size();

    cuda_graph_data->edge_data_len = edge_data_list->size();

    cuda_graph_data->adj_vertex_list_len = adj_vertex_id_list->size();

    vertex_id_type* cuda_vertex_id_list = NULL;
    vertex_data_type* cuda_vertex_data_list = NULL;

    size_t* cuda_edge_id_list = NULL;
    edge_data_type* cuda_edge_data_list = NULL;
    
    vertex_id_type* cuda_adj_vertex_id_list = NULL;
    vertex_data_type* cuda_adj_data_list = NULL;

    cudaMalloc((void**)&cuda_vertex_id_list,  get_vertex_id_list_byte_len(cuda_graph_data));
    cudaMalloc((void**)&cuda_vertex_data_list,  get_vertex_data_list_byte_len(cuda_graph_data));
    cudaMalloc((void**)&cuda_edge_id_list,  get_edge_id_list_byte_len(cuda_graph_data));
    cudaMalloc((void**)&cuda_edge_data_list,  get_edge_data_list_byte_len(cuda_graph_data));
    cudaMalloc((void**)&cuda_adj_vertex_id_list,  get_adj_vertex_id_list_byte_len(cuda_graph_data));
    cudaMalloc((void**)&cuda_adj_data_list,  get_adj_vertex_data_list_byte_len(cuda_graph_data));
    // ret gather
    cudaMalloc((void**)&cuda_graph_data->gather_nums_ingpu,  get_gather_num_byte_len(cuda_graph_data));



    cudaMemcpy(cuda_vertex_id_list,vertex_index_list->data(), get_vertex_id_list_byte_len(cuda_graph_data),cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_vertex_data_list,vertex_data_list->data(), get_vertex_data_list_byte_len(cuda_graph_data),cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_edge_id_list,edge_id_list->data(), get_edge_id_list_byte_len(cuda_graph_data),cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_edge_data_list,edge_data_list->data(), get_edge_data_list_byte_len(cuda_graph_data),cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_adj_vertex_id_list,adj_vertex_id_list->data(), get_adj_vertex_id_list_byte_len(cuda_graph_data),cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_adj_data_list,adj_data_list->data(), get_adj_vertex_data_list_byte_len(cuda_graph_data),cudaMemcpyHostToDevice);

    if(cuda_graph_data->GPU_EXCUTE_APPLY_READY){
        cudaMemcpy(cuda_graph_data->gather_nums_ingpu,(cuda_graph_data->apply_gather_nums_incpu)->data(),
                     get_gather_num_byte_len(cuda_graph_data),cudaMemcpyHostToDevice);

    }

    cuda_graph_data->vertex_id_list_ingpu = cuda_vertex_id_list;
    cuda_graph_data->vertex_data_list_ingpu = cuda_vertex_data_list;
    cuda_graph_data->edge_id_list_ingpu = cuda_edge_id_list;
    cuda_graph_data->edge_data_list_ingpu = cuda_edge_data_list;
    cuda_graph_data->adj_vertex_id_list_ingpu = cuda_adj_vertex_id_list;
    cuda_graph_data->adj_vertex_data_list_ingpu = cuda_adj_data_list;

}
/**
 * \brief release graph in gpu for float
 * \param cuda_graph_data Graph data stored in GPU 
*/
template<> void cuda_graph_free<cuda_graph_data_type<float, float, float>*>(cuda_graph_data_type<float, float, float>* cuda_graph_data)
{
    cudaFree(cuda_graph_data->vertex_id_list_ingpu);
    cudaFree(cuda_graph_data->vertex_data_list_ingpu);
    cudaFree(cuda_graph_data->edge_id_list_ingpu);
    cudaFree(cuda_graph_data->edge_data_list_ingpu);
    cudaFree(cuda_graph_data->adj_vertex_id_list_ingpu);
    cudaFree(cuda_graph_data->adj_vertex_data_list_ingpu);
    delete (cuda_graph_data->vertex_data_list_incpu);
    delete (cuda_graph_data->edge_id_list_incpu);
    delete (cuda_graph_data->edge_data_list_incpu);
    delete (cuda_graph_data->adj_vertex_id_list_incpu);
    delete (cuda_graph_data->adj_vertex_data_list_incpu);
    delete (cuda_graph_data->gather_nums_incpu);
}
/**
 * \brief release graph in gpu
 * \param cuda_graph_data Graph data stored in GPU 
*/
template<typename cuda_graph_type>
void cuda_graph_free(cuda_graph_type cuda_graph_data)
{
    cudaFree(cuda_graph_data->vertex_id_list_ingpu);
    cudaFree(cuda_graph_data->vertex_data_list_ingpu);
    cudaFree(cuda_graph_data->edge_id_list_ingpu);
    cudaFree(cuda_graph_data->edge_data_list_ingpu);
    cudaFree(cuda_graph_data->adj_vertex_id_list_ingpu);
    cudaFree(cuda_graph_data->adj_vertex_data_list_ingpu);
    delete (cuda_graph_data->vertex_data_list_incpu);
    delete (cuda_graph_data->edge_id_list_incpu);
    delete (cuda_graph_data->edge_data_list_incpu);
    delete (cuda_graph_data->adj_vertex_id_list_incpu);
    delete (cuda_graph_data->adj_data_list_incpu);
    delete (cuda_graph_data->gather_nums_incpu);
}

// The vertex data is just the cuda_pagerank value (a float)
typedef float vertex_data_type;

// There is no edge data in the cuda_pagerank application
typedef float edge_data_type;

typedef struct cuda_graph_data_type<float,float,float> graph_data_type;
/**
 * \brief excute gather with gpu
 * \param vertex_id vertex id
 * \param graph_data graph data in gpu
 * \return gather value
*/
__device__ float cuda_gather(size_t vertex_id, graph_data_type* graph_data){
    int nums_out_edges = (graph_data->edge_id_list_ingpu)[vertex_id+1] - (graph_data->edge_id_list_ingpu)[vertex_id];
    int RESET_PROB_VALUE = 0.15;
    return  ( (1.0-RESET_PROB_VALUE) / nums_out_edges ) * (graph_data->vertex_data_list_ingpu)[vertex_id];
}
/**
 * \brief excute apply with gpu
 * \param vertex_id vertex id
 * \param graph_data graph data in gpu
 * \return gather value
*/
__device__ float cuda_apply(size_t vertex_id, graph_data_type* graph_data){
    int RESET_PROB_VALUE = 0.15;
    double newval = graph_data->apply_gather_nums_ingpu[vertex_id] + RESET_PROB_VALUE;
    float last_change = std::fabs(newval - graph_data->vertex_data_list_ingpu[vertex_id]);
    //graph_data->vertex_data_list_ingpu[vertex_id] = newval;
    return newval;
}

// typedef float(*vertex_program_gather_func)(size_t,cuda_graph_data_type<float, float, float>*);
// __global__ void cuda_excute_gather_inner(cuda_graph_data_type<float, float, float>*  cuda_graph_data,
//                                          vertex_program_gather_func vertex_program_gather)
// {
//     size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if(idx>=cuda_graph_data->vertex_list_len) return;
//     //size_t vertex_id = (cuda_graph_data->vertex_id_list_ingpu)[idx];
//     //T5 gather_num = vertex_program_gather(vertex_id,cuda_graph_data);
//     float gather_num = vertex_program_gather(idx,cuda_graph_data);
//     (cuda_graph_data->gather_nums_ingpu)[idx] = gather_num;

// }
/**
 * \brief excute gather with gpu by gather
 * \param cuda_graph_data graph stored in gpu
 * \param vertex_program_gather vertex program that user defined
*/
template<typename T1, class T4, typename T5 = float>
__global__ void cuda_excute_gather_inner( T1 cuda_graph_data,
                                        T4 vertex_program_gather)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=cuda_graph_data->vertex_list_len) return;
    //size_t vertex_id = (cuda_graph_data->vertex_id_list_ingpu)[idx];
    //T5 gather_num = vertex_program_gather(vertex_id,cuda_graph_data);
    T5 gather_num = vertex_program_gather(idx,cuda_graph_data);
    (cuda_graph_data->gather_nums_ingpu)[idx] = gather_num;

}

/**
 * \brief excute gather with gpu
 * \param cuda_graph_data graph stored in gpu
*/
template<typename cuda_graph_type>
void cuda_excute_gather(cuda_graph_type cuda_graph_data)
{
    size_t thread_count = 1024;
    //TODO HAS QUESTION MAX_BLOCK=1024
    // ALTHOUGH vertex_list_len < 1024
    size_t block_count = cuda_graph_data->vertex_list_len/thread_count + 1;
	

    cuda_excute_gather_inner<<<block_count,thread_count>>>(
                            cuda_graph_data,
                            cuda_gather
                            );
    //void* gather_value = malloc(get_gather_num_byte_len(cuda_graph_data));
    std::vector<float>* gather_value = new std::vector<float>(get_gather_num_byte_len(cuda_graph_data));
    cudaMemcpy(gather_value->data(), cuda_graph_data->gather_nums_ingpu,
        get_gather_num_byte_len(cuda_graph_data),cudaMemcpyDeviceToHost);
    cuda_graph_data->gather_nums_incpu = gather_value;

}



void cuda_sync_gather_host()
{

}
//cuda_graph_data_type<float, float, float> *, void (pagerank::*)(size_t, pagerank::graph_data_type *)
/**
 * \brief excute apply with gpu by apply
 * \param cuda_graph_data graph stored in gpu
 * \param vertex_program_apply vertex program that user defined
*/
template<typename T1, class T4, typename T5 = float>
__global__ void cuda_excute_apply_inner( T1 cuda_graph_data,
                                        T4 vertex_program_apply)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=cuda_graph_data->vertex_list_len) return;
    
    T5 apply_num = vertex_program_apply(idx,cuda_graph_data);
    (cuda_graph_data->apply_gather_nums_ingpu)[idx] = apply_num;
}

/**
 * \brief excute apply with gpu
 * \param cuda_graph_data graph stored in gpu
*/
template<typename cuda_graph_type>
void cuda_excute_apply(cuda_graph_type cuda_graph_data)
{
    size_t thread_count = 1024;
    size_t block_count = cuda_graph_data->vertex_list_len/thread_count + 1;
	

    cuda_excute_apply_inner<<<block_count,thread_count>>>(
                            cuda_graph_data,
                            cuda_apply
                            );
    //void* apply_value = malloc(get_gather_num_byte_len(cuda_graph_data));
    //typedef decltype(*(cuda_graph_data->gather_nums_incpu)) apply_data_type;
    typedef std::vector<float> apply_data_type;
    apply_data_type* apply_value = new apply_data_type(cuda_graph_data->vertex_list_len,0);
    
    cudaMemcpy(apply_value->data(), cuda_graph_data->apply_gather_nums_ingpu,
        get_adj_vertex_data_list_byte_len(cuda_graph_data),cudaMemcpyDeviceToHost);
    cuda_graph_data->gather_nums_incpu = apply_value;
}


// cuda sync apply
void cuda_sync_apply()
{

}
// cuda scatter
void cuda_excute_scatter()
{

}
#include "dc_cuda_interface.cu"

#endif
#endif