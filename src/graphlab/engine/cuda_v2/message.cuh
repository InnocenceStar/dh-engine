#include<vector>
namespace cuda{
template<typename gather_num_type>
struct gather_messages{
	gather_num_type* gather_nums = NULL;
};
template<typename gather_num_type>
struct apply_messages{
	gather_num_type* apply_nums = NULL;
};
}
namespace host{
template<typename gather_num_type>
struct gather_messages{
	std::vector<gather_num_type> gather_nums;
};
template<typename gather_num_type>
struct apply_messages{
	std::vector<gather_num_type> apply_nums;
};
}