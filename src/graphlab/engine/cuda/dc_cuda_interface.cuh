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
#define GPU
#ifdef GPU
#ifndef DC_CUDA_INTERFACE_CUH
#define DC_CUDA_INTERFACE_CUH
#include <cuda_runtime.h>
#include <graphlab/engine/cuda/cuda_ivertex_program.cuh>
#include <graphlab/vertex_program/icontext.hpp>
#include <graphlab/vertex_program/context.hpp>
#include <graphlab/graph/distributed_graph.hpp>

/**
 * \brief gpu status in interation
 * \param GPU_EXCUTE_GATHER_READY ready to execute gather
 * \param GPU_EXCUTE_GATHER_DONE done executing gather
 * \param GPU_EXCUTE_APPLY_READY ready to execute apply
 * \param GPU_EXCUTE_APPLY_DONE done executing apply
 * \param GPU_EXCUTE_SCATTER_READY ready to execute scatter
 * \param GPU_EXCUTE_SCATTER_DONE done executing scatter
*/
enum GPU_STATUS{
	GPU_EXCUTE_GATHER_READY = 0,
	GPU_EXCUTE_GATHER_DONE,
	GPU_EXCUTE_APPLY_READY,
	GPU_EXCUTE_APPLY_DONE,
	GPU_EXCUTE_SCATTER_READY,
	GPU_EXCUTE_SCATTER_DONE
};
/**
 * \brief data structure for managing gpu graph data
 * \param gpu_status gpu status
 * \param vertex_id_list_incpu vertex id stored in cpu
 * \param vertex_id_list_ingpu vertex id stored in gpu
 * \param vertex_data_list_incpu vertex data stored in cpu
 * \param vertex_data_list_ingpu vertex data stored in gpu
 * \param vertex_list_len the number of vertex
 * \param edge_id_list_incpu edge id stored in cpu
 * \param edge_id_list_ingpu edge id stored in gpu
 * \param edge_data_list_incpu edge data stored in cpu
 * \param edge_data_list_ingpu edge data stored in gpu
 * \param edge_list_len the number of edge
 * \param adj_vertex_id_list_incpu adj vertex id stored in cpu
 * \param adj_vertex_id_list_ingpu adj vertex id stored in gpu
 * \param adj_vertex_data_list_incpu adj vertex data stored in cpu
 * \param adj_vertex_data_list_ingpu adj vertex data stored in gpu
 * \param adj_vertex_list_len the number of adj vertex
 * \param gather_nums_ingpu gather num stored in gpu
 * \param gather_nums_incpu gather num stored in cpu
 * \param apply_gather_nums_ingpu apply num stored in gpu
 * \param apply_gather_nums_incpu apply num stored in cpu
*/
template<class T1,class T2,class T3>
struct cuda_graph_data_type
{
	enum GPU_STATUS gpu_status = GPU_EXCUTE_GATHER_DONE;//maybe use
	std::vector<size_t, std::allocator<size_t>>* vertex_id_list_incpu = NULL;
	size_t* vertex_id_list_ingpu = NULL;
	std::vector<T1, std::allocator<T1>>* vertex_data_list_incpu = NULL;
	T1* vertex_data_list_ingpu = NULL;
	int vertex_list_len = 0;

    size_t* edge_id_list_ingpu = NULL;
	std::vector<size_t, std::allocator<size_t>>* edge_id_list_incpu = NULL;
	int edge_index_len = 0;
	
	T2* edge_data_list_ingpu = NULL;
	std::vector<T2, std::allocator<T2>>* edge_data_list_incpu = NULL;
	int edge_data_len = 0;

	std::vector<size_t, std::allocator<size_t>>* adj_vertex_id_list_incpu = NULL;
	size_t* adj_vertex_id_list_ingpu = NULL;

    std::vector<T1, std::allocator<T1>>* adj_vertex_data_list_incpu = NULL;
	T1* adj_vertex_data_list_ingpu = NULL;
	int adj_vertex_list_len = 0;

	T3* gather_nums_ingpu = NULL;
	std::vector<T3, std::allocator<T3>>* gather_nums_incpu = NULL;
	// T3 gather_total = 0;

	T3* apply_gather_nums_ingpu = NULL;
	std::vector<T3, std::allocator<T3>>* apply_gather_nums_incpu = NULL;

};

/**
 * @brief get vertex id byte
 * @tparam CUDA_GRAPH_DATA 
 * @param a graph stored in gpu
 * @return size of vertex id
 */
template<class CUDA_GRAPH_DATA>
size_t get_vertex_id_list_byte_len(CUDA_GRAPH_DATA* a)
{
	return a->vertex_list_len * sizeof(size_t);
};
/**
 * @brief get vertex data byte
 * @tparam CUDA_GRAPH_DATA 
 * @param a graph stored in gpu
 * @return size of vertex data
 */
template<class CUDA_GRAPH_DATA>
size_t get_vertex_data_list_byte_len(CUDA_GRAPH_DATA* a)
{
	return a->vertex_list_len * sizeof(*(a->vertex_data_list_incpu));
};
template<class CUDA_GRAPH_DATA>
size_t get_gather_num_byte_len(CUDA_GRAPH_DATA* a)
{
	return a->vertex_list_len * sizeof(*(a->gather_nums_incpu));
};
/**
 * @brief get edge id byte
 * @tparam CUDA_GRAPH_DATA 
 * @param a graph stored in gpu
 * @return size of edge id
 */
template<class CUDA_GRAPH_DATA>
size_t get_edge_id_list_byte_len(CUDA_GRAPH_DATA* a)
{
	return a->edge_index_len * sizeof(size_t);
};
/**
 * @brief get edge data byte
 * @tparam CUDA_GRAPH_DATA 
 * @param a graph stored in gpu
 * @return size of edge data
 */
template<class CUDA_GRAPH_DATA>
size_t get_edge_data_list_byte_len(CUDA_GRAPH_DATA* a)
{
	return a->edge_data_len * sizeof(*(a->edge_data_list_incpu));
};
/**
 * @brief get adj vertex id byte
 * @tparam CUDA_GRAPH_DATA 
 * @param a graph stored in gpu
 * @return size of edj vertex id
 */
template<class CUDA_GRAPH_DATA>
size_t get_adj_vertex_id_list_byte_len(CUDA_GRAPH_DATA* a)
{
	return a->adj_vertex_list_len * sizeof(size_t);
};
/**
 * @brief get adj vertex data byte
 * @tparam CUDA_GRAPH_DATA 
 * @param a graph stored in gpu
 * @return size of adj vertex data
 */
template<class CUDA_GRAPH_DATA>
size_t get_adj_vertex_data_list_byte_len(CUDA_GRAPH_DATA* a)
{
	return a->adj_vertex_list_len * sizeof(*(a->adj_vertex_data_list_incpu));
};

/**
 * cuda env init
 * @param null
 * @return the num of gpu
 * */
int cuda_env_init();


// cuda graph compute init,include data assginment
/**
 * \brief Explicitly specify cuda data initialization
 * \param vertex_index_list Graph vertex index list executed by GPU
 * \param graph local graph
 * \param cuda_graph_data Graph data stored in GPU 
*/
template<typename Cuda_Vertex,typename Graph_Type,typename vertex_graph_type>
void cuda_data_init(Cuda_Vertex vertex, Graph_Type& graph,vertex_graph_type *cuda_graph_data);

/**
 * \brief release graph in gpu for float
 * \param cuda_graph_data Graph data stored in GPU 
*/
template<typename cuda_graph_type>
void cuda_graph_free(cuda_graph_type cuda_graph_data);
// cuda gather 
/**
 * \brief excute gather with gpu by gather
 * \param cuda_graph_data graph stored in gpu
 * \param vertex_program_gather vertex program that user defined
*/
template<typename cuda_graph_type>
void cuda_excute_gather(cuda_graph_type cuda_graph_data);

// cuda apply
/**
 * \brief excute apply with gpu by apply
 * \param cuda_graph_data graph stored in gpu
 * \param vertex_program_apply vertex program that user defined
*/
template<typename cuda_graph_type>
void cuda_excute_apply(cuda_graph_type cuda_graph_data);

void cuda_sync_apply_host();
// cuda scatter
void cuda_excute_scatter();


#include "./dc_cuda_interface.cu"

#endif
#endif