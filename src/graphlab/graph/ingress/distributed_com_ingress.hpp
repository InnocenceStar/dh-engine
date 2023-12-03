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

#ifndef DHE_DISTRIBUTED_COM_INGRESS_HPP
#define DHE_DISTRIBUTED_COM_INGRESS_HPP

#include <boost/functional/hash.hpp>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <cmath> 

#include <graphlab/rpc/buffered_exchange.hpp>
#include <graphlab/graph/graph_basic_types.hpp>
#include <graphlab/graph/ingress/distributed_ingress_base.hpp>
#include <graphlab/graph/distributed_graph.hpp>
#include <graphlab/graph/graph_hash.hpp>
#include <graphlab/rpc/buffered_exchange.hpp>
#include <graphlab/rpc/distributed_event_log.hpp>
#include <graphlab/parallel/pthread_tools.hpp>
#include <graphlab/logger/logger.hpp>
#include <graphlab/macros_def.hpp>
#include <graphlab/util/system_info.hpp>

#define QAQ

namespace graphlab
{
	template <typename VertexData, typename EdgeData>
	class distributed_graph;

	/**
	 * \brief Ingress object assigning edges
	 * 		using a distributed collaborative stream partitioning method.
	 */
	template <typename VertexData, typename EdgeData>
	class distributed_dc_ingress : public distributed_ingress_base<VertexData, EdgeData>
	{

	public:
		// the type of distributed_graph object for communication
		typedef distributed_graph<VertexData, EdgeData> graph_type;
		// The type of the vertex data stored in the graph
		typedef VertexData vertex_data_type;
		// The type of the edge data stored in the graph
		typedef EdgeData edge_data_type;
		// base type of distributed ingress
		typedef distributed_ingress_base<VertexData, EdgeData> base_type;
		// the type of the vertex record which stores information
		// (global vertex id,degree type,owner,num of incoming edges and outgoing edges
		//  vertex mirrors exclude owner)
		// associated with each vertex on this proc
		typedef typename graph_type::vertex_record vertex_record;
		// the type of mirror : fixed_dense_bitset
		typedef typename graph_type::mirror_type mirror_type;

		// the type which can cache vertex id in buffer for sending
		typedef typename buffered_exchange<vertex_id_type>::buffer_type
				vertex_id_buffer_type;
		// the type of buffer which include source id,target id and edge data;
		typedef typename base_type::edge_buffer_record edge_buffer_record;
		//  the type which can cache edge in buffer for sending
		typedef typename buffered_exchange<edge_buffer_record>::buffer_type
				edge_buffer_type;
		// the type which include vertex id,vertex data;
		typedef typename base_type::vertex_buffer_record vertex_buffer_record;
		// the type of buffer which contains vertex_buffer_record(vertex id,vertex data)
		typedef typename buffered_exchange<vertex_buffer_record>::buffer_type
				vertex_buffer_type;
		// the type which is map that each element contains vertex id
		// and correspanding vector of edge_buffer_record (source id,target id and edge data)
		typedef typename boost::unordered_map<vertex_id_type,
																					std::vector<edge_buffer_record>>
				raw_map_type;

		// detail vertex record for the second pass coordination.
		// which contains mirrors, num of in edges and out edges, bool value for having data,vertex data
		typedef typename base_type::vertex_negotiator_record
				vertex_negotiator_record;

		// hash table contains vertex id and its master location
		typedef typename boost::unordered_map<vertex_id_type, procid_t>
				master_hash_table_type;
		// belong to above hash table
		typedef typename std::pair<vertex_id_type, procid_t>
				master_pair_type;
		// comm with other machines ,send by master_pair_type
		typedef typename buffered_exchange<master_pair_type>::buffer_type
				master_buffer_type;

		// pair contains proc id and corresponding score
		typedef typename std::pair<procid_t, size_t>
				proc_score_pair_type;
		// use to send above pair
		typedef typename buffered_exchange<proc_score_pair_type>::buffer_type
				proc_score_buffer_type;

		/// The rpc interface for this object
		dc_dist_object<distributed_dc_ingress> dc_rpc;
		/// The underlying distributed graph object that is being loaded
		graph_type &graph;

		// threshold to divide high-degree and low-degree vertices
		size_t threshold;

		bool standalone;
		// the list store edge_buffer_record(source id,target id and edge data)
		std::vector<edge_buffer_record> dc_edges;

		/* ingress exchange */
		// send edge_buffer_record to other machines
		buffered_exchange<edge_buffer_record> dc_edge_exchange;
		// send vertex_buffer_record( vertex id,vertex data) to other machines
		buffered_exchange<vertex_buffer_record> dc_vertex_exchange;

		// send high degree vertex's edge_buffer_record to other machines
		buffered_exchange<edge_buffer_record> high_edge_exchange;
		// send low degree vertex's edge_buffer_record to other machines
		buffered_exchange<edge_buffer_record> low_edge_exchange;
		// resend vertex_buffer_record( vertex id,vertex data) to other machines
		buffered_exchange<vertex_buffer_record> resend_vertex_exchange;

		/** master hash table (mht): location mapping of low-degree vertices  */
		master_hash_table_type mht;
		buffered_exchange<master_pair_type> mht_exchange;

		// consider both edge and vertex
		std::vector<size_t> proc_balance;
		// the increment for proc score
		std::vector<size_t> proc_score_incr;
		// use to send porc and corresponding score
		buffered_exchange<proc_score_pair_type> proc_score_exchange;
		
		/*
		// unused now. use later
		// pair to store proc vertex_num
		typedef typename std::pair<graphlab::procid_t, size_t> proc_vertex_num;

		// store current vertex of proc
		std::vector<size_t> proc_related_vertex_num;
		buffered_exchange<proc_vertex_num> proc_vertex_num_exchange;
		*/

		/// heuristic model from fennel
		/// records about the number of edges and vertices in the graph
		/// given from the commandline
		size_t tot_nedges;
		size_t tot_nverts;
		/// threshold for incremental mht to be synced across the cluster
		/// when the incremental mht size reaches the preset interval,
		/// we will perform a synchronization on mht across the cluster
		size_t interval;
		// arguments for the algorithm
		double alpha;
		double gamma;

	public:
		distributed_dc_ingress(distributed_control &dc, graph_type &graph,
													 size_t threshold = 100, size_t tot_nedges = 0, size_t tot_nverts = 0,
													 size_t interval = std::numeric_limits<size_t>::max()) : base_type(dc, graph), dc_rpc(dc, this),
														graph(graph), threshold(threshold),
#ifdef _OPENMP
														dc_edge_exchange(dc, omp_get_max_threads()),
														dc_vertex_exchange(dc, omp_get_max_threads()),
#else
														dc_edge_exchange(dc),
														dc_vertex_exchange(dc),
#endif
														high_edge_exchange(dc), low_edge_exchange(dc), resend_vertex_exchange(dc),
														mht_exchange(dc), proc_balance(dc.numprocs()),
														proc_score_incr(dc.numprocs()), proc_score_exchange(dc),
														tot_nedges(tot_nedges), tot_nverts(tot_nverts), interval(interval)
		{
			ASSERT_GT(tot_nedges, 0);
			ASSERT_GT(tot_nverts, 0);

			gamma = 1.5;
			alpha = sqrt(dc.numprocs()) * double(tot_nedges) / pow(tot_nverts, gamma);

			/* fast pass for standalone case. */
			standalone = dc_rpc.numprocs() == 1;
			dc_rpc.barrier();
		} // end of constructor

		~distributed_dc_ingress() {}

		/** Add an edge to the ingress object using random hashing assignment.
		 *  This function acts as the first phase for TSV graph to deliver edges
		 *  via the hashing value of its target vertex.
		 */
		void add_edge(vertex_id_type source, vertex_id_type target,
									const EdgeData &edata)
		{
			const edge_buffer_record record(source, target, edata);
			const procid_t owning_proc = standalone ? 0 : graph_hash::hash_vertex(target) % dc_rpc.numprocs();
#ifdef _OPENMP
			dc_edge_exchange.send(owning_proc, record, omp_get_thread_num());
#else
			dc_edge_exchange.send(owning_proc, record);
#endif
		} // end of add edge

		/* add vdata */
		void add_vertex(vertex_id_type vid, const VertexData &vdata)
		{
			const vertex_buffer_record record(vid, vdata);
			const procid_t owning_proc = standalone ? 0 : graph_hash::hash_vertex(vid) % dc_rpc.numprocs();
#ifdef _OPENMP
			dc_vertex_exchange.send(owning_proc, record, omp_get_thread_num());
#else
			dc_vertex_exchange.send(owning_proc, record);
#endif
		} // end of add vertex

		/* dc heuristic for low-degree vertex */
		procid_t dc_to_proc(const vertex_id_type target,
												const std::vector<edge_buffer_record> &records)
		{
			size_t nprocs = dc_rpc.numprocs();
			std::vector<double> proc_score(nprocs);
			std::vector<int> proc_degrees(nprocs);

			// get vertex nums of every proc
			// temporarily use degrees num to instead vertex num
			for (size_t i = 0; i < records.size(); ++i)
			{
				if (mht.find(records[i].source) != mht.end()) // mht:{key:vertex id,value:owner proc id}
					proc_degrees[mht[records[i].source]]++;
			}
			// get the coll level value
			//  get cpu load
			//  temporarily use cpu utilization rate
			float cpu_usage = graphlab::systeminfo::get_cpu_usage();

			// get gpu load
			// temporarily use gpu utilization rate
			float gpu_usage = graphlab::systeminfo::get_gpu_usage();

			float cor = (cpu_usage > gpu_usage) ? (cpu_usage - gpu_usage) : (gpu_usage - cpu_usage);

			// get the partation control value
			// tot_nedges
			int avg_degrees = tot_nedges / nprocs;
			std::vector<int> part_fault(nprocs);
			for (size_t i = 0; i < nprocs; ++i)
			{
				int temp_degrees_value = proc_degrees[i] / avg_degrees;
				if (part_fault.size() == 0)
				{
					part_fault.push_back(temp_degrees_value);
				}
			}

			// compute the score to get the best partition
			for (size_t i = 0; i < nprocs; ++i)
			{
				proc_score[i] = proc_degrees[i] * pow(2,cor) * (1-part_fault[i]);
			}

			double best_score = proc_score[0];
			procid_t best_proc = 0;
			for (size_t i = 1; i < nprocs; ++i)
			{
				if (proc_score[i] > best_score)
				{
					best_score = proc_score[i];
					best_proc = i;
				}
			}

			return best_proc;
		};

		/* dc heuristic for low-degree vertex */
		void sync_heuristic()
		{
			size_t nprocs = dc_rpc.numprocs();
			procid_t l_procid = dc_rpc.procid();

			// send proc_score_incr
			for (procid_t p = 0; p < nprocs; p++)
			{
				for (procid_t i = 0; i < nprocs; i++)
					if (i != l_procid)
						proc_score_exchange.send(i, std::make_pair(p, proc_score_incr[p]));
				proc_score_incr[p] = 0;
			}

			// flush proc_score_incr and mht but w/o spin
			proc_score_exchange.partial_flush(0);
			mht_exchange.partial_flush(0);

			// update local mht and proc_balance
			master_buffer_type master_buffer;
			procid_t proc = -1;
			while (mht_exchange.recv(proc, master_buffer, false))
			{
				foreach (const master_pair_type &pair, master_buffer)
					// update vertex location
					mht[pair.first] = pair.second;
			}
			mht_exchange.clear();

			proc_score_buffer_type proc_edge_buffer;
			proc = -1;
			while (proc_score_exchange.recv(proc, proc_edge_buffer, false))
			{
				foreach (const proc_score_pair_type &pair, proc_edge_buffer)
					proc_balance[pair.first] += pair.second;
			}
			proc_score_exchange.clear();
		}

		/**
		 * @brief assgin edges by dc
		 * @param null
		 * @return null
		*/
		void assign_dc_edges()
		{
			graphlab::timer ti;
			size_t nprocs = dc_rpc.numprocs();
			procid_t l_procid = dc_rpc.procid();
			//target vertex id and related edge_buffer_record array
			raw_map_type raw_map;
			size_t vcount = 0;

			// collect edges
			edge_buffer_type edge_buffer;
			procid_t proc = -1;
			while (dc_edge_exchange.recv(proc, edge_buffer))
			{
				foreach (const edge_buffer_record &rec, edge_buffer)
				{
					raw_map[rec.target].push_back(rec);
				}
			}
			dc_edge_exchange.clear();

			if (l_procid == 0)
			{
				logstream(LOG_INFO) << "collect raw map: "
														<< ti.current_time()
														<< " secs"
														<< std::endl;
			}
			logstream(LOG_INFO) << "receive " << raw_map.size()
													<< " vertices done." << std::endl;

			// assign vertices and its in-edges to hosting node
			for (typename raw_map_type::iterator it = raw_map.begin();
					 it != raw_map.end(); ++it)
			{
				vertex_id_type target = it->first;
				procid_t owning_proc = 0;
				size_t degree = it->second.size();
				// high degree assgin strategy
				if (degree > threshold)
				{
					// TODO: no need send, just resend latter
					owning_proc = graph_hash::hash_vertex(target) % nprocs;
					for (size_t i = 0; i < degree; ++i)
						high_edge_exchange.send(owning_proc, it->second[i]);
				}
				else
				{
					// low degree assgin by using dc_to_proc func
					owning_proc = dc_to_proc(target, it->second);
					for (size_t i = 0; i < degree; ++i)
						low_edge_exchange.send(owning_proc, it->second[i]);

					// update mht and nedges_incr
					for (procid_t p = 0; p < nprocs; ++p)
					{
						if (p != l_procid)
							mht_exchange.send(p, master_pair_type(target, owning_proc));
						else
							mht[target] = owning_proc;
					}

					// adjust balance according to vertex and edge
					proc_balance[owning_proc]++;
					proc_balance[owning_proc] +=
							(degree * float(tot_nverts) / float(tot_nedges));

					proc_score_incr[owning_proc]++;
					proc_score_incr[owning_proc] +=
							(degree * float(tot_nverts) / float(tot_nedges));
				}

				// periodical synchronize heurisitic
				if ((++vcount % interval) == 0)
					sync_heuristic();
			}

			// last synchronize on mht
			mht_exchange.flush();
			// vector to store vertex_id-location data
			master_buffer_type master_buffer;
			proc = -1;
			while (mht_exchange.recv(proc, master_buffer))
			{
				foreach (const master_pair_type &pair, master_buffer)
					mht[pair.first] = pair.second;
			}
			mht_exchange.clear();

			// logstream(LOG_INFO) << "balance[";
			// for (procid_t i = 0; i < nprocs; i++)
			//   logstream(LOG_INFO) << proc_balance[i] << ",";
			// logstream(LOG_INFO) << "] ";
			logstream(LOG_INFO) << "nsyncs(" << (vcount / interval)
													<< ") using " << ti.current_time() << " secs "
													<< "#mht=" << mht.size()
													<< std::endl;
		}

		/**
		 * @brief opti and sync and clean
		*/
		void finalize()
		{
			graphlab::timer ti;

			size_t nprocs = dc_rpc.numprocs();
			procid_t l_procid = dc_rpc.procid();
			size_t nedges = 0;

			dc_rpc.full_barrier();

			if (l_procid == 0)
			{
				memory_info::log_usage("start finalizing");
				logstream(LOG_EMPH) << "dc finalizing ..."
														<< " #vertices=" << graph.local_graph.num_vertices()
														<< " #edges=" << graph.local_graph.num_edges()
														<< " threshold=" << threshold
														<< " interval=" << interval
														<< " gamma=" << gamma
														<< " alpha=" << alpha
														<< std::endl;
			}

			/**************************************************************************/
			/*                                                                        */
			/*                       Flush any additional data                        */
			/*                                                                        */
			/**************************************************************************/
			dc_edge_exchange.flush();
			dc_vertex_exchange.flush();

			/**
			 * Fast pass for redundant finalization with no graph changes.
			 */
			{
				size_t changed_size = dc_edge_exchange.size() + dc_vertex_exchange.size();
				dc_rpc.all_reduce(changed_size);
				if (changed_size == 0)
				{
					logstream(LOG_INFO) << "Skipping Graph Finalization because no changes happened..." << std::endl;
					return;
				}
			}

			/**************************************************************************/
			/*                                                                        */
			/*                             Assign edges                             */
			/*                                                                        */
			/**************************************************************************/
			if (!standalone)
				assign_dc_edges();

			if (l_procid == 0)
			{
				logstream(LOG_INFO) << "assign edges: "
														<< ti.current_time()
														<< " secs"
														<< std::endl;
			}

			/**************************************************************************/
			/*                                                                        */
			/*                       Prepare dc ingress                           */
			/*                                                                        */
			/**************************************************************************/
			if (standalone)
			{ /* fast pass for standalone */
				edge_buffer_type edge_buffer;
				procid_t proc = -1;
				nedges = dc_edge_exchange.size();

				while (dc_edge_exchange.recv(proc, edge_buffer))
				{
					foreach (const edge_buffer_record &rec, edge_buffer)
						dc_edges.push_back(rec);
				}
				dc_edge_exchange.clear();
			}
			else
			{
				high_edge_exchange.flush();
				low_edge_exchange.flush();

				nedges = low_edge_exchange.size();
				dc_edges.reserve(nedges + high_edge_exchange.size());

				edge_buffer_type edge_buffer;
				procid_t proc = -1;
				// assgin unprocessed low edges
				while (low_edge_exchange.recv(proc, edge_buffer))
				{
					foreach (edge_buffer_record &rec, edge_buffer)
					{
						if (mht.find(rec.source) == mht.end())
							mht[rec.source] = graph_hash::hash_vertex(rec.source) % nprocs;

						dc_edges.push_back(rec);
					}
				}
				low_edge_exchange.clear();

				if (l_procid == 0)
				{
					logstream(LOG_INFO) << "low-degree edges: "
															<< ti.current_time()
															<< " secs"
															<< std::endl;
				}

				proc = -1;
				// assgin unprocessed high edges
				while (high_edge_exchange.recv(proc, edge_buffer))
				{
					foreach (const edge_buffer_record &rec, edge_buffer)
					{
						if (mht.find(rec.source) == mht.end())
							mht[rec.source] = graph_hash::hash_vertex(rec.source) % nprocs;

						const procid_t owner_proc = mht[rec.source];
						if (owner_proc == l_procid)
						{
							dc_edges.push_back(rec);
							++nedges;
						}
						else
						{
							// re-send edges of high-degree vertices by dc_edge_exchange
							dc_edge_exchange.send(owner_proc, rec);
						}
					}
				}
				high_edge_exchange.clear();

				// receive edges of high-degree vertices
				dc_edge_exchange.flush();
				logstream(LOG_INFO) << "receive #edges=" << dc_edge_exchange.size()
														<< std::endl;
				proc = -1;
				while (dc_edge_exchange.recv(proc, edge_buffer))
				{
					foreach (const edge_buffer_record &rec, edge_buffer)
					{
						mht[rec.source] = l_procid;
						dc_edges.push_back(rec);
						++nedges;
					}
				}
				dc_edge_exchange.clear();
			}



			if (l_procid == 0)
			{
				memory_info::log_usage("prepare dc finalizing done.");
				logstream(LOG_EMPH) << "prepare dc finalizing. ("
														<< ti.current_time()
														<< " secs)"
														<< std::endl;
			}

			// connect to base finalize()
			modified_base_finalize(nedges);

			// set vertex degree type for dc engine
			set_degree_type();

			if (l_procid == 0)
			{
				logstream(LOG_EMPH) << "dc finalizing graph. ("
														<< ti.current_time()
														<< " secs)"
														<< std::endl;
			}
		} // end of finalize

		void set_degree_type()
		{
			graphlab::timer ti;
			procid_t l_procid = dc_rpc.procid();
			size_t high_master = 0, high_mirror = 0, low_master = 0, low_mirror = 0;

			for (size_t lvid = 0; lvid < graph.num_local_vertices(); lvid++)
			{
				vertex_record &vrec = graph.lvid2record[lvid];
				if (vrec.num_in_edges > threshold)
				{
					vrec.dtype = graph_type::HIGH;
					if (vrec.owner == l_procid)
						high_master++;
					else
						high_mirror++;
				}
				else
				{
					vrec.dtype = graph_type::LOW;
					if (vrec.owner == l_procid)
						low_master++;
					else
						low_mirror++;
				}
			}

#ifdef TUNING
			// Compute the total number of high-degree and low-degree vertices
			std::vector<size_t> swap_counts(dc_rpc.numprocs());

			swap_counts[l_procid] = high_master;
			dc_rpc.all_gather(swap_counts);
			high_master = 0;
			foreach (size_t count, swap_counts)
				high_master += count;

			swap_counts[l_procid] = high_mirror;
			dc_rpc.all_gather(swap_counts);
			high_mirror = 0;
			foreach (size_t count, swap_counts)
				high_mirror += count;

			swap_counts[l_procid] = low_master;
			dc_rpc.all_gather(swap_counts);
			low_master = 0;
			foreach (size_t count, swap_counts)
				low_master += count;

			swap_counts[l_procid] = low_mirror;
			dc_rpc.all_gather(swap_counts);
			low_mirror = 0;
			foreach (size_t count, swap_counts)
				low_mirror += count;

			if (l_procid == 0)
			{
				logstream(LOG_EMPH) << "dc info: master ["
														<< high_master << " "
														<< low_master << " "
														<< (float(high_master) / (high_master + low_master)) << "]"
														<< std::endl;
				if ((high_mirror + low_mirror) > 0)
					logstream(LOG_EMPH) << "dc info: mirror ["
															<< high_mirror << " "
															<< low_mirror << " "
															<< (float(high_mirror) / (high_mirror + low_mirror)) << "]"
															<< std::endl;

				memory_info::log_usage("set vertex type done.");
				logstream(LOG_EMPH) << "set vertex type: "
														<< ti.current_time()
														<< " secs"
														<< std::endl;
			}
#endif
		}

		/* do the same job as original base finalize except for
		 * extracting edges from dc_edges instead of original edge_buffer;
		 * and using mht to tracing the master location of each vertex.
		 */
		void modified_base_finalize(size_t nedges)
		{
			graphlab::timer ti;
			procid_t l_procid = dc_rpc.procid();
			size_t nprocs = dc_rpc.numprocs();

			dc_rpc.full_barrier();

			bool first_time_finalize = false;
			/**
			 * Fast pass for first time finalization.
			 */
			if (graph.is_dynamic())
			{
				size_t nverts = graph.num_local_vertices();
				dc_rpc.all_reduce(nverts);
				first_time_finalize = (nverts == 0);
			}
			else
			{
				first_time_finalize = false;
			}

			typedef typename hopscotch_map<vertex_id_type, lvid_type>::value_type
					vid2lvid_pair_type;

			/**
			 * \internal
			 * Buffer storage for new vertices to the local graph.
			 */
			typedef typename graph_type::hopscotch_map_type vid2lvid_map_type;
			vid2lvid_map_type vid2lvid_buffer;

			/**
			 * \internal
			 * The begining id assinged to the first new vertex.
			 */
			const lvid_type lvid_start = graph.vid2lvid.size();

			/**
			 * \internal
			 * Bit field incidate the vertex that is updated during the ingress.
			 */
			dense_bitset updated_lvids(graph.vid2lvid.size());

			/**************************************************************************/
			/*                                                                        */
			/*                         Construct local graph                          */
			/*                                                                        */
			/**************************************************************************/
			{ // Add all the edges to the local graph
				graph.local_graph.reserve_edge_space(nedges + 1);

				foreach (const edge_buffer_record &rec, dc_edges)
				{
					// Get the source_vlid;
					lvid_type source_lvid(-1);
					if (graph.vid2lvid.find(rec.source) == graph.vid2lvid.end())
					{
						if (vid2lvid_buffer.find(rec.source) == vid2lvid_buffer.end())
						{
							source_lvid = lvid_start + vid2lvid_buffer.size();
							vid2lvid_buffer[rec.source] = source_lvid;
						}
						else
						{
							source_lvid = vid2lvid_buffer[rec.source];
						}
					}
					else
					{
						source_lvid = graph.vid2lvid[rec.source];
						updated_lvids.set_bit(source_lvid);
					}
					// Get the target_lvid;
					lvid_type target_lvid(-1);
					if (graph.vid2lvid.find(rec.target) == graph.vid2lvid.end())
					{
						if (vid2lvid_buffer.find(rec.target) == vid2lvid_buffer.end())
						{
							target_lvid = lvid_start + vid2lvid_buffer.size();
							vid2lvid_buffer[rec.target] = target_lvid;
						}
						else
						{
							target_lvid = vid2lvid_buffer[rec.target];
						}
					}
					else
					{
						target_lvid = graph.vid2lvid[rec.target];
						updated_lvids.set_bit(target_lvid);
					}
					graph.local_graph.add_edge(source_lvid, target_lvid, rec.edata);
				} // end for loop over buffers
				dc_edges.clear();

				ASSERT_EQ(graph.vid2lvid.size() + vid2lvid_buffer.size(),
									graph.local_graph.num_vertices());
#ifdef TUNING
				if (l_procid == 0)
				{
					logstream(LOG_INFO) << "populating local graph: "
															<< ti.current_time()
															<< " secs"
															<< std::endl;
				}
#endif
				// Finalize local graph
				graph.local_graph.finalize();
#ifdef TUNING
				logstream(LOG_INFO) << "local graph info: " << std::endl
														<< "\t nverts: " << graph.local_graph.num_vertices()
														<< std::endl
														<< "\t nedges: " << graph.local_graph.num_edges()
														<< std::endl;

				if (l_procid == 0)
				{
					logstream(LOG_INFO) << "finalizing local graph: "
															<< ti.current_time()
															<< " secs"
															<< std::endl;
				}
#endif
			}

			/**************************************************************************/
			/*                                                                        */
			/*             Receive and add vertex data to masters                     */
			/*                                                                        */
			/**************************************************************************/
			// Setup the map containing all the vertices being negotiated by this machine
			{
				if (standalone)
				{
					vertex_buffer_type vertex_buffer;
					procid_t proc = -1;
					while (dc_vertex_exchange.recv(proc, vertex_buffer))
					{
						foreach (const vertex_buffer_record &rec, vertex_buffer)
						{
							lvid_type lvid(-1);
							if (graph.vid2lvid.find(rec.vid) == graph.vid2lvid.end())
							{
								if (vid2lvid_buffer.find(rec.vid) == vid2lvid_buffer.end())
								{
									lvid = lvid_start + vid2lvid_buffer.size();
									vid2lvid_buffer[rec.vid] = lvid;
								}
								else
								{
									lvid = vid2lvid_buffer[rec.vid];
								}
							}
							else
							{
								lvid = graph.vid2lvid[rec.vid];
								updated_lvids.set_bit(lvid);
							}
							if (distributed_dc_ingress::vertex_combine_strategy && lvid < graph.num_local_vertices())
							{
								distributed_dc_ingress::vertex_combine_strategy(
										graph.l_vertex(lvid).data(), rec.vdata);
							}
							else
							{
								graph.local_graph.add_vertex(lvid, rec.vdata);
							}
						}
					}
					dc_vertex_exchange.clear();
				}
				else
				{
					// re-send by trampoline
					vertex_buffer_type vertex_buffer;
					procid_t proc = -1;
					while (dc_vertex_exchange.recv(proc, vertex_buffer))
					{
						foreach (const vertex_buffer_record &rec, vertex_buffer)
						{
							if (mht.find(rec.vid) == mht.end())
								mht[rec.vid] = graph_hash::hash_vertex(rec.vid) % nprocs;
							resend_vertex_exchange.send(mht[rec.vid], rec);
						}
					}
					dc_vertex_exchange.clear();

					// receive vertex data re-sent by other machines
					resend_vertex_exchange.flush();
					proc = -1;
					while (resend_vertex_exchange.recv(proc, vertex_buffer))
					{
						foreach (const vertex_buffer_record &rec, vertex_buffer)
						{
							lvid_type lvid(-1);
							if (graph.vid2lvid.find(rec.vid) == graph.vid2lvid.end())
							{
								if (vid2lvid_buffer.find(rec.vid) == vid2lvid_buffer.end())
								{
									lvid = lvid_start + vid2lvid_buffer.size();
									vid2lvid_buffer[rec.vid] = lvid;
								}
								else
								{
									lvid = vid2lvid_buffer[rec.vid];
								}
							}
							else
							{
								lvid = graph.vid2lvid[rec.vid];
								updated_lvids.set_bit(lvid);
							}
							if (distributed_dc_ingress::vertex_combine_strategy && lvid < graph.num_local_vertices())
							{
								distributed_dc_ingress::vertex_combine_strategy(
										graph.l_vertex(lvid).data(), rec.vdata);
							}
							else
							{
								graph.local_graph.add_vertex(lvid, rec.vdata);
							}
						}
					}
					resend_vertex_exchange.clear();
				}

#ifdef TUNING
				if (l_procid == 0)
				{
					logstream(LOG_INFO) << "adding vertex data: "
															<< ti.current_time()
															<< " secs"
															<< std::endl;
				}
#endif
			} // end of loop to populate vrecmap

			/**************************************************************************/
			/*                                                                        */
			/*        Assign vertex data and allocate vertex (meta)data  space        */
			/*                                                                        */
			/**************************************************************************/
			{
				// determine masters for all negotiated vertices
				const size_t local_nverts = graph.vid2lvid.size() + vid2lvid_buffer.size();
				graph.lvid2record.reserve(local_nverts);
				graph.lvid2record.resize(local_nverts);
				graph.local_graph.resize(local_nverts);
				foreach (const vid2lvid_pair_type &pair, vid2lvid_buffer)
				{
					vertex_record &vrec = graph.lvid2record[pair.second];
					vrec.gvid = pair.first;
					if (standalone)
					{
						vrec.owner = 0;
					}
					else
					{
						if (mht.find(pair.first) == mht.end())
							mht[pair.first] = graph_hash::hash_vertex(pair.first) % nprocs;
						vrec.owner = mht[pair.first];
					}
				}
				ASSERT_EQ(local_nverts, graph.local_graph.num_vertices());
				ASSERT_EQ(graph.lvid2record.size(), graph.local_graph.num_vertices());
#ifdef TUNING
				if (l_procid == 0)
				{
					logstream(LOG_INFO) << "allocating lvid2record: "
															<< ti.current_time()
															<< " secs"
															<< std::endl;
				}
#endif
				mht.clear();
			}

			/**************************************************************************/
			/*                                                                        */
			/*                          Master handshake                              */
			/*                                                                        */
			/**************************************************************************/
			if (!standalone)
			{
#ifdef _OPENMP
				buffered_exchange<vertex_id_type> vid_buffer(dc_rpc.dc(), omp_get_max_threads());
#else
				buffered_exchange<vertex_id_type> vid_buffer(dc_rpc.dc());
#endif

#ifdef _OPENMP
#pragma omp parallel for
#endif
				// send not owned vids to their master
				for (lvid_type i = lvid_start; i < graph.lvid2record.size(); ++i)
				{
					procid_t master = graph.lvid2record[i].owner;
					if (master != l_procid)
#ifdef _OPENMP
						vid_buffer.send(master, graph.lvid2record[i].gvid, omp_get_thread_num());
#else
						vid_buffer.send(master, graph.lvid2record[i].gvid);
#endif
				}
				vid_buffer.flush();
				dc_rpc.barrier();

				// receive all vids owned by me
				mutex flying_vids_lock;
				boost::unordered_map<vertex_id_type, mirror_type> flying_vids;
#ifdef _OPENMP
#pragma omp parallel
#endif
				{
					typename buffered_exchange<vertex_id_type>::buffer_type buffer;
					procid_t recvid = -1;
					while (vid_buffer.recv(recvid, buffer))
					{
						foreach (const vertex_id_type vid, buffer)
						{
							if (graph.vid2lvid.find(vid) == graph.vid2lvid.end())
							{
								if (vid2lvid_buffer.find(vid) == vid2lvid_buffer.end())
								{
									flying_vids_lock.lock();
									mirror_type &mirrors = flying_vids[vid];
									mirrors.set_bit(recvid);
									flying_vids_lock.unlock();
								}
								else
								{
									lvid_type lvid = vid2lvid_buffer[vid];
									graph.lvid2record[lvid]._mirrors.set_bit(recvid);
								}
							}
							else
							{
								lvid_type lvid = graph.vid2lvid[vid];
								graph.lvid2record[lvid]._mirrors.set_bit(recvid);
								updated_lvids.set_bit(lvid);
							}
						}
					}
				}
				vid_buffer.clear();

				if (!flying_vids.empty())
				{
					logstream(LOG_INFO) << "#flying-own-nverts="
															<< flying_vids.size()
															<< std::endl;

					// reallocate spaces for the flying vertices.
					size_t vsize_old = graph.lvid2record.size();
					size_t vsize_new = vsize_old + flying_vids.size();
					graph.lvid2record.resize(vsize_new);
					graph.local_graph.resize(vsize_new);
					for (typename boost::unordered_map<vertex_id_type, mirror_type>::iterator it = flying_vids.begin();
							 it != flying_vids.end(); ++it)
					{
						lvid_type lvid = lvid_start + vid2lvid_buffer.size();
						vertex_record &vrec = graph.lvid2record[lvid];
						vertex_id_type gvid = it->first;
						vrec.owner = l_procid;
						vrec.gvid = gvid;
						vrec._mirrors = it->second;
						vid2lvid_buffer[gvid] = lvid;
					}
				}
			} // end of master handshake

#ifdef TUNING
			if (l_procid == 0)
			{
				logstream(LOG_INFO) << "master handshake: "
														<< ti.current_time()
														<< " secs"
														<< std::endl;
			}
#endif

			/**************************************************************************/
			/*                                                                        */
			/*                        Merge in vid2lvid_buffer                        */
			/*                                                                        */
			/**************************************************************************/
			{
				if (graph.vid2lvid.size() == 0)
				{
					graph.vid2lvid.swap(vid2lvid_buffer);
				}
				else
				{
					graph.vid2lvid.rehash(graph.vid2lvid.size() + vid2lvid_buffer.size());
					foreach (const typename vid2lvid_map_type::value_type &pair, vid2lvid_buffer)
					{
						graph.vid2lvid.insert(pair);
					}
					vid2lvid_buffer.clear();
				}
			}

			/**************************************************************************/
			/*                                                                        */
			/*              Synchronize vertex data and meta information              */
			/*                                                                        */
			/**************************************************************************/
			// TODO:  optimization for standalone
			{
				// construct the vertex set of changed vertices

				// Fast pass for first time finalize;
				vertex_set changed_vset(true);

				// Compute the vertices that needs synchronization
				if (!first_time_finalize)
				{
					vertex_set changed_vset = vertex_set(false);
					changed_vset.make_explicit(graph);
					updated_lvids.resize(graph.num_local_vertices());
					for (lvid_type i = lvid_start; i < graph.num_local_vertices(); ++i)
					{
						updated_lvids.set_bit(i);
					}
					changed_vset.localvset = updated_lvids;
					buffered_exchange<vertex_id_type> vset_exchange(dc_rpc.dc());
					// sync vset with all mirrors
					changed_vset.synchronize_mirrors_to_master_or(graph, vset_exchange);
					changed_vset.synchronize_master_to_mirrors(graph, vset_exchange);
				}

				graphlab::graph_gather_apply<graph_type, vertex_negotiator_record>
						vrecord_sync_gas(graph,
														 boost::bind(&distributed_dc_ingress::finalize_gather, this, _1, _2),
														 boost::bind(&distributed_dc_ingress::finalize_apply, this, _1, _2, _3));
				vrecord_sync_gas.exec(changed_vset);

#ifdef TUNING
				if (l_procid == 0)
				{
					logstream(LOG_INFO) << "synchrionizing vertex (meta)data: "
															<< ti.current_time()
															<< " secs"
															<< std::endl;
				}
#endif
			}

			base_type::exchange_global_info(standalone);
#ifdef TUNING
			if (l_procid == 0)
			{
				logstream(LOG_INFO) << "exchange global info: "
														<< ti.current_time()
														<< " secs"
														<< std::endl;
			}
#endif

			if (l_procid == 0)
			{
				memory_info::log_usage("base finalizing done.");
				logstream(LOG_EMPH) << "base finalizing. ("
														<< ti.current_time()
														<< " secs)"
														<< std::endl;
			}
		} // end of modified base finalize

	private:
		boost::function<void(vertex_data_type &, const vertex_data_type &)> vertex_combine_strategy;

		/**
		 * \brief Gather the vertex distributed meta data.
		 */
		vertex_negotiator_record finalize_gather(lvid_type &lvid, graph_type &graph)
		{
			vertex_negotiator_record accum;
			accum.num_in_edges = graph.local_graph.num_in_edges(lvid);
			accum.num_out_edges = graph.local_graph.num_out_edges(lvid);
			if (graph.l_is_master(lvid))
			{
				accum.has_data = true;
				accum.vdata = graph.l_vertex(lvid).data();
				accum.mirrors = graph.lvid2record[lvid]._mirrors;
			}
			return accum;
		}

		/**
		 * \brief Update the vertex data structures with the gathered vertex metadata.
		 */
		void finalize_apply(lvid_type lvid, const vertex_negotiator_record &accum, graph_type &graph)
		{
			typename graph_type::vertex_record &vrec = graph.lvid2record[lvid];
			vrec.num_in_edges = accum.num_in_edges;
			vrec.num_out_edges = accum.num_out_edges;
			graph.l_vertex(lvid).data() = accum.vdata;
			vrec._mirrors = accum.mirrors;
		}
	}; // end of distributed_dc_ingress
};	 // end of namespace graphlab
#include <graphlab/macros_undef.hpp>

#endif