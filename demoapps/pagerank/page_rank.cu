#include <vector>
#include <string>
#include <fstream>

#include <graphlab.hpp>
#include <graphlab/engine/cuda/dc_cuda_interface.cuh>
#include <graphlab/engine/cuda/cuda_dc_engine.cuh>
#include <graphlab/engine/cuda/cuda_ivertex_program.cuh>
#include <graphlab/engine/cuda/cuda_omni_engine.cuh>
// #include <graphlab/macros_def.hpp>

// Global random reset probability
float RESET_PROB_VALUE = 0.15;

float TOLERANCE_VALUE = 1.0E-2;

// // The vertex data is just the cuda_pagerank value (a float)
// typedef float vertex_data_type;

// // There is no edge data in the cuda_pagerank application
// typedef float edge_data_type;

// typedef struct cuda_graph_data_type<float,float,float> graph_data_type;
// The graph type is determined by the vertex and edge data types
typedef graphlab::distributed_graph<vertex_data_type, edge_data_type> graph_type;

// template<typename T1,typename T2,typename T3>
// struct cuda_graph_data_type
// {
// 	size_t* vertex_id_list = NULL;
// 	T1* vertex_data_list = NULL;
// 	int vertex_list_len = 0;

// 	T3* gather_nums = NULL;

//     size_t* edge_id_list_incpu = NULL;
// 	int edge_index_len = 0;
	
// 	T2* edge_data_list = NULL;
// 	int edge_data_len = 0;

// 	size_t* adj_vertex_id_list = NULL;
//     T1* adj_vertex_list_data_list = NULL;
// 	int adj_vertex_list_len = 0;
// };

// typedef struct cuda_graph_data_type<vertex_data_type, void, float> graph_data_type;
/*
 * A simple function used by graph.transform_vertices(init_vertex_func);
 * to initialize the vertes data.
 */
void init_vertex_func(graph_type::vertex_type& vertex) { vertex.data() = 1; }


class cuda_pagerank :
  public graphlab::cuda_ivertex_program<graph_type, float>,
  public graphlab::IS_POD_TYPE {
  float last_change;
public:
  /* Gather the weighted rank of the adjacent page   */
  float gather(icontext_type& context, const vertex_type& vertex,
               edge_type& edge) const {
    return ((1.0 - RESET_PROB_VALUE) / edge.source().num_out_edges()) *
      edge.source().data();
  }
 
  /* Use the total rank of adjacent pages to update this page */
  void apply(icontext_type& context, vertex_type& vertex,
             const gather_type& total) {
    const double newval = total + RESET_PROB_VALUE;
    last_change = std::fabs(newval - vertex.data());
    vertex.data() = newval;
  }


  /* The scatter edges depend on whether the cuda_pagerank has converged */
  edge_dir_type scatter_edges(icontext_type& context,
                              const vertex_type& vertex) const {
    if (last_change > TOLERANCE_VALUE) return graphlab::OUT_EDGES;
    else return graphlab::NO_EDGES;
  }

  /* The scatter function just signal adjacent pages */
  void scatter(icontext_type& context, const vertex_type& vertex,
               edge_type& edge) const {
    context.signal(edge.target());
  }
}; // end of factorized_pagerank update functor


/*
 * We want to save the final graph so we define a write which will be
 * used in graph.save("path/prefix", cuda_pagerank_writer()) to save the graph.
 */
struct cuda_pagerank_writer {
  std::string save_vertex(graph_type::vertex_type v) {
    std::stringstream strm;
    strm << v.id() << "\t" << v.data() << "\n";
    return strm.str();
  }
  std::string save_edge(graph_type::edge_type e) { return ""; }
}; // end of cuda_pagerank writer



int main(int argc, char** argv) {
  // Initialize control plain using mpi
  graphlab::mpi_tools::init(argc, argv);
  graphlab::distributed_control dc;
  global_logger().set_log_level(LOG_INFO);

  // Parse command line options -----------------------------------------------
  graphlab::command_line_options clopts("cuda_pagerank algorithm.");
  std::string graph_dir;
  std::string format = "adj";
  std::string exec_type = "synchronous";
  clopts.attach_option("graph", graph_dir,
                       "The graph file. Required ");
  clopts.add_positional("graph");
  clopts.attach_option("format", format,
                       "The graph file format");
  clopts.attach_option("engine", exec_type, 
                       "The engine type synchronous or asynchronous");
  clopts.attach_option("tol", TOLERANCE_VALUE,
                       "The permissible change at convergence.");
  std::string saveprefix;
  clopts.attach_option("saveprefix", saveprefix,
                       "If set, will save the resultant cuda_pagerank to a "
                       "sequence of files with prefix saveprefix");

  if(!clopts.parse(argc, argv)) {
    dc.cout() << "Error in parsing command line arguments." << std::endl;
    return EXIT_FAILURE;
  }

  if (graph_dir == "") {
    dc.cout() << "Graph not specified. Cannot continue";
    return EXIT_FAILURE;
  }

  // Build the graph ----------------------------------------------------------
  graph_type graph(dc, clopts);
  dc.cout() << "Loading graph in format: "<< format << std::endl;
  graph.load_format(graph_dir, format);
  // must call finalize before querying the graph
  graph.finalize();
  dc.cout() << "#vertices: " << graph.num_vertices()
            << " #edges:" << graph.num_edges() << std::endl;

  // Initialize the vertex data
  graph.transform_vertices(init_vertex_func);

  // Running The Engine -------------------------------------------------------
  graphlab::cuda_omni_engine<cuda_pagerank> engine(dc, graph, exec_type, clopts);
  engine.signal_all();
  engine.start();
  const float runtime = engine.elapsed_seconds();
  dc.cout() << "Finished Running engine in " << runtime
            << " seconds." << std::endl;

  // Save the final graph -----------------------------------------------------
  if (saveprefix != "") {
    graph.save(saveprefix, cuda_pagerank_writer(),
               false,    // do not gzip
               true,     // save vertices
               false);   // do not save edges
  }

  // Tear-down communication layer and quit -----------------------------------
  graphlab::mpi_tools::finalize();
  return EXIT_SUCCESS;
} // End of main