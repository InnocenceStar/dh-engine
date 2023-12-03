#include<cuda_runtime.h>
#include<vector>
namespace cuda{
/**
 * @brief local grpah for device
 * @tparam vertex_data_type 
 * @tparam edge_data_type 
 */
template<typename vertex_data_type,typename edge_data_type>
struct local_graph{
	size_t* vertex_id_list = NULL;
	vertex_data_type* vertex_data_list = NULL;
	size_t vertex_nums = 0;

	size_t* edge_id_list = NULL;
	edge_data_type* edge_data_list = NULL;
	size_t edge_nums = 0;

	size_t* adjvertex_id_list = NULL;
	vertex_data_type* adjvertex_data_list = NULL;
	size_t adjvertex_nums = 0;
};

}

namespace cuda{

}


namespace host{
/**
 * @brief local graph buffer for host
 * @tparam vertex_data_type 
 * @tparam edge_data_type 
 */
template<typename vertex_data_type,typename edge_data_type>
struct local_graph_buffer{
	typedef vertex_data_type vertex_data_type;
	typedef edge_data_type edge_data_type;

	//size_t* vertex_id_list = NULL;
	std::vector<size_t> vertex_id_list;
	//vertex_data_type* vertex_data_list = NULL;
	std::vector<vertex_data_type> vertex_data_list;
	size_t vertex_nums = 0;

	//size_t* edge_id_list = NULL;
	std::vector<size_t> edge_id_list;
	//edge_data_type* edge_data_list = NULL;
	std::vector<edge_data_type> edge_data_list;
	size_t edge_nums = 0;

	//size_t* adjvertex_id_list = NULL;
	std::vector<size_t> adjvertex_id_list;
	//vertex_data_type* adjvertex_data_list = NULL;
	std::vector<vertex_data_type> adjvertex_data_list;
	size_t adjvertex_nums = 0;
	
	void set_vertex_nums(size_t nums);
	const size_t get_vertex_idl_byte();
	const size_t get_vertex_datal_byte();

	void set_edge_nums(size_t nums);
	const size_t get_edge_idl_byte();
	const size_t get_edge_datal_byte();

	void set_adjvertex_nums(size_t nums);
	const size_t get_adjvertex_idl_byte();
	const size_t get_adjvertex_datal_byte();
	void post_init_lgraph_buffer();

};

}

namespace host{
template<typename vertex_data_type,typename edge_data_type>
void local_graph_buffer<vertex_data_type,edge_data_type>::
set_vertex_nums(size_t num){
	this->vertex_nums = num;
}
template<typename vertex_data_type,typename edge_data_type>
void local_graph_buffer<vertex_data_type,edge_data_type>::
set_edge_nums(size_t num){
	this->edge_nums = num;
}
template<typename vertex_data_type,typename edge_data_type>
void local_graph_buffer<vertex_data_type,edge_data_type>::
set_adjvertex_nums(size_t num){
	this->adjvertex_nums = num;
}

template<typename vertex_data_type,typename edge_data_type>
const size_t local_graph_buffer<vertex_data_type,edge_data_type>::
get_vertex_idl_byte(){
	return this->vertex_nums * sizeof(size_t);
}
template<typename vertex_data_type,typename edge_data_type>
const size_t local_graph_buffer<vertex_data_type,edge_data_type>::
get_vertex_datal_byte(){
	return this->vertex_nums * sizeof(vertex_data_type);
}
template<typename vertex_data_type,typename edge_data_type>
const size_t local_graph_buffer<vertex_data_type,edge_data_type>::
get_edge_idl_byte(){
	return this->edge_nums * sizeof(size_t);
}
template<typename vertex_data_type,typename edge_data_type>
const size_t local_graph_buffer<vertex_data_type,edge_data_type>::
get_edge_datal_byte(){
	return this->edge_nums * sizeof(edge_data_type);
}
template<typename vertex_data_type,typename edge_data_type>
const size_t local_graph_buffer<vertex_data_type,edge_data_type>::
get_adjvertex_idl_byte(){
	return this->adjvertex_nums * sizeof(size_t);
}
template<typename vertex_data_type,typename edge_data_type>
const size_t local_graph_buffer<vertex_data_type,edge_data_type>::
get_adjvertex_datal_byte(){
	return this->adjvertex_nums * sizeof(vertex_data_type);
}

template<typename vertex_data_type,typename edge_data_type>
void local_graph_buffer<vertex_data_type,edge_data_type>::
post_init_lgraph_buffer(){
	set_vertex_nums(this->vertex_id_list.size());
	set_edge_nums(this->edge_id_list.size());
	set_adjvertex_nums(this->adjvertex_id_list.size());
}

}

