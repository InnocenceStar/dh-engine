project(GraphLab)


add_graphlab_executable(simple_pagerank simple_pagerank.cpp)

add_graphlab_executable(cuda_pagerank page_rank.cu)

add_graphlab_executable(warp_parfor_pagerank warp_parfor_pagerank.cpp)

add_graphlab_executable(warp_engine_pagerank warp_engine_pagerank.cpp)
