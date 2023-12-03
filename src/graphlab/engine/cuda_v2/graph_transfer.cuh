#include<cuda_runtime.h>
#include<vector>
#include <boost/foreach.hpp>
#include"graph_cuda.cuh"

namespace host{

/**
 * @brief init local graph buffer
 * if u want use ,u need 显示声明类型
 * @tparam graph_type 
 * @tparam local_graph_buffer_type need provided
 * @param local_graph 
 * @param vertex_id_list 
 * @return  local_graph_buffer
 */
template<typename graph_type, typename local_graph_buffer_type>
local_graph_buffer_type* init_local_graph_buffer(graph_type& graph,
							std::vector<size_t>& vertex_id_list){

    typedef typename graph_type::vertex_data_type    vertex_data_type;
    typedef typename graph_type::edge_data_type    edge_data_type;
    typedef typename graph_type::local_vertex_type    local_vertex_type;
    typedef typename graph_type::local_edge_type    local_edge_type;
    typedef typename graph_type::vertex_type          vertex_type;
    typedef typename graph_type::edge_type            edge_type;


	local_graph_buffer_type* local_graphbuf = 
			new local_graph_buffer_type();
	local_graphbuf->vertex_id_list = vertex_id_list;
	int edgeidx_perver = 0;
	for(auto vertex_id = vertex_id_list.begin();
		vertex_id!=vertex_id_list.end();
		++vertex_id)
	{
		local_vertex_type local_vertex = graph.l_vertex(*vertex_id);
		local_graphbuf->vertex_data_list.push_back(local_vertex.data());

		BOOST_FOREACH(local_edge_type local_edge, local_vertex.in_edges()) {
            ++index;
            edge_type edge(local_edge);
            local_graphbuf->edge_data_list.push_back(edge.data());
            local_graphbuf->adjvertex_id_list.push_back(edge.source().id());
            local_graphbuf->adjvertex_data_list.push_back(edge.source().data());


        }

		local_graphbuf->edge_id_list.push_back(edgeidx_perver);

		local_graphbuf->post_init_lgraph_buffer();
	}

}


}

namespace cuda{
/**
 * @brief 申请cuda本地图存储空间
 * @tparam local_graph_buffer_type 穿入未init的指针
 * @tparam local_graph_type 
 * @param local_gbuf 
 * @return 
 */
template<typename local_graph_buffer_type,typename local_graph_type>
local_graph_type* create_lgraph(local_graph_buffer_type* local_gbuf){
	typedef typename local_graph_buffer_type::vertex_data_type vertex_data_type;
	typedef typename local_graph_buffer_type::edge_data_type edge_data_type;
	local_graph_type* local_graph = 
		new local_graph_type();
	cudaMalloc((void**)&local_graph->vertex_id_list, local_gbuf->get_vertex_idl_byte());
    cudaMalloc((void**)&local_graph->vertex_data_list, local_gbuf->get_vertex_datal_byte());
    cudaMalloc((void**)&local_graph->edge_id_list, local_gbuf->get_edge_idl_byte());
    cudaMalloc((void**)&local_graph->edge_data_list, local_gbuf->get_edge_datal_byte());
    cudaMalloc((void**)&local_graph->adjvertex_id_list, local_gbuf->get_adjvertex_idl_byte());
    cudaMalloc((void**)&local_graph->adjvertex_data_list, local_gbuf->get_adjvertex_datal_byte());
}
/**
 * @brief 初始化cuda本地图
 * @tparam local_graph_buffer_type 
 * @tparam local_graph_type 
 * @param local_gbuf 
 * @return 
 */
template<typename local_graph_buffer_type,typename local_graph_type>
void init_lgraph(local_graph_type* cuda_local_graph,local_graph_buffer_type* local_gbuf){
	typedef typename local_graph_buffer_type::vertex_data_type vertex_data_type;
	typedef typename local_graph_buffer_type::edge_data_type edge_data_type;
	cudaMemcpy(cuda_local_graph->vertex_id_list,local_gbuf->vertex_id_list.data(),
				local_gbuf->get_vertex_idl_byte(),cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_local_graph->vertex_data_list,local_gbuf->vertex_data_list.data(),
				local_gbuf->get_vertex_datal_byte(),cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_local_graph->edge_id_list,local_gbuf->edge_id_list.data(),
				local_gbuf->get_edge_idl_byte(),cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_local_graph->edge_data_list,local_gbuf->edge_data_list.data(),
				local_gbuf->get_edge_datal_byte(),cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_local_graph->adjvertex_id_list,local_gbuf->adjvertex_id_list.data(),
				local_gbuf->get_adjvertex_idl_byte(),cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_local_graph->adjvertex_data_list,local_gbuf->adjvertex_data_list.data(),
				local_gbuf->get_adjvertex_datal_byte(),cudaMemcpyHostToDevice);
}

/**
 * @brief 清除图空间数据，不释放空间
 * @tparam local_graph_type 
 * @param cuda_local_graph 
 * @param local_gbuf 
 */
template<typename local_graph_type>
void clear_lgraph(local_graph_type* cuda_local_graph){
	cudaMemset(cuda_local_graph,0,cuda_local_graph->vertex_nums);
	cuda_local_graph->vertex_nums = 0;
	
}
/**
 * @brief 释放图空间
 * @tparam local_graph_buffer_type 
 * @tparam local_graph_type 
 * @param cuda_local_graph 
 * @param local_gbuf 
 */
template<typename local_graph_buffer_type,typename local_graph_type>
void destory_lgraph(local_graph_type* cuda_local_graph,local_graph_buffer_type* local_gbuf){
	typedef typename local_graph_buffer_type::vertex_data_type vertex_data_type;
	typedef typename local_graph_buffer_type::edge_data_type edge_data_type;
	cudaFree(cuda_local_graph->vertex_id_list_ingpu);
    cudaFree(cuda_local_graph->vertex_data_list_ingpu);
    cudaFree(cuda_local_graph->edge_id_list_ingpu);
    cudaFree(cuda_local_graph->edge_data_list_ingpu);
    cudaFree(cuda_local_graph->adj_vertex_id_list_ingpu);
    cudaFree(cuda_local_graph->adj_vertex_data_list_ingpu);
}

}