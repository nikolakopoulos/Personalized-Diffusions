///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 Contains the PERDIF program that implements personalized diffusions for recommendations.

 Code by: Dimitris Berberidis and Athanasios N. Nikolakopoulos
 University of Minnesota 2019
*/

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>
#include <stdbool.h>
#include <pthread.h>

#include "rec_IO.h"
#include "rec_defs.h"
#include "rec_mem.h"
#include "csr_handling.h"
#include "rec_graph_gen.h"

static csr_graph_t glue(user_t *users, csr_graph_t M_I, sz_long_t num_users);

//Creates the large graph were the actual random walks will be run
csr_graph_t generate_rec_graph(user_t *users, csr_graph_t item_model, sz_long_t num_users, bool bipartite)
{
	csr_graph_t graph;

	if (bipartite)
	{
		csr_graph_t M_I = csr_transpose(item_model);
		csr_rec_normalize(&M_I);

		//Glue together and also take into account teleportation prob.

		graph = glue(users, M_I, num_users);

		csr_free(M_I);
	}
	else
	{
		graph = csr_transpose(item_model);
		csr_rec_normalize(&graph);
	}

	return graph;
}

//Glues together R, R^T, and normalized W^T
static csr_graph_t glue(user_t *users, csr_graph_t M_I, sz_long_t num_users)
{

	//Allocate
	csr_graph_t graph;
	csr_alloc(&graph);

	sz_long_t num_ratings = 0;
	for (sz_long_t i = 0; i < num_users; i++)
		num_ratings += users[i].num_items;

	csr_realloc(&graph, 2 * num_ratings + num_users + M_I.nnz, M_I.num_nodes + num_users);

	// Prepare items (essentially R^T)
	int itm_buff_size = num_users;
	item_t *items = (item_t *)malloc(M_I.num_nodes * sizeof(item_t));
	for (sz_long_t i = 0; i < M_I.num_nodes; i++)
	{
		items[i].users = (sz_long_t *)malloc(itm_buff_size * sizeof(sz_long_t));
		items[i].num_users = 0;
	}

	for (sz_long_t i = 0; i < num_users; i++)
	{
		//printf("%d \n",(int)i);
		for (sz_long_t j = 0; j < users[i].num_items; j++)
		{
			items[users[i].items[j]].num_users += 1;
			if (items[users[i].items[j]].num_users > itm_buff_size)
			{
				//printf("%d \n",(int)items[users[i].items[j]].num_users);
				sz_long_t *temp_item = (sz_long_t *)malloc((items[users[i].items[j]].num_users - 1) * sizeof(sz_long_t));
				memcpy(temp_item, items[users[i].items[j]].users, items[users[i].items[j]].num_users * sizeof(sz_long_t));
				items[users[i].items[j]].users = realloc(items[users[i].items[j]].users, items[users[i].items[j]].num_users * sizeof(sz_long_t));
				memcpy(items[users[i].items[j]].users, temp_item, items[users[i].items[j]].num_users * sizeof(sz_long_t));
				free(temp_item);
			}
			items[users[i].items[j]].users[items[users[i].items[j]].num_users - 1] = i;
		}
	}

	if (DEBUG)
	{
		for (sz_long_t i = 0; i < M_I.num_nodes; i++)
		{
			for (sz_long_t j = 0; j < items[i].num_users; j++)
			{
				printf("%d ", (int)items[i].users[j]);
			}
			printf("\n\n");
		}
		printf("%d %d \n", (int)M_I.num_nodes, (int)num_users);
	}

	for (sz_long_t i = 0; i < M_I.num_nodes; i++)
	{
		//printf("%d \n",(int) items[i].num_users );
		items[i].users = realloc(items[i].users, items[i].num_users * sizeof(sz_long_t));
	}

	//Calculate normalizers once
	double *item_inv_freq = (double *)malloc(M_I.num_nodes * sizeof(double));
	double *usr_inv_freq = (double *)malloc(num_users * sizeof(double));
	for (sz_long_t i = 0; i < M_I.num_nodes; i++)
		item_inv_freq[i] = 1.0 / (double)items[i].num_users;
	for (sz_long_t i = 0; i < num_users; i++)
		usr_inv_freq[i] = 1.0 / (double)users[i].num_items;

	// Populate graph with subgraphs ( R and R^T normalized on the fly)

	//First N rows

	double alpha = ALPHA;

	graph.csr_row_pointer[0] = 0;
	sz_long_t edge_counter = 0;
	for (sz_long_t i = 0; i < num_users; i++)
	{
		//Add row of R with normalized entries
		for (sz_long_t j = 0; j < users[i].num_items; j++)
		{
			graph.csr_column[edge_counter] = users[i].items[j] + num_users;
			graph.csr_value[edge_counter] = alpha * item_inv_freq[users[i].items[j]];
			edge_counter++;
		}
		//Add diagonal element
		graph.csr_column[edge_counter] = i;
		graph.csr_value[edge_counter] = 1.0 - alpha;
		edge_counter++;
		//Advance row pointer
		graph.csr_row_pointer[i + 1] = edge_counter;
	}

	//Remaining M rows
	for (sz_long_t i = num_users; i < graph.num_nodes; i++)
	{

		sz_long_t k = i - num_users; // Local counter of M_I nodes

		//Add row of R^T with normalized entries
		for (sz_long_t j = 0; j < items[k].num_users; j++)
		{
			graph.csr_column[edge_counter] = items[k].users[j];
			graph.csr_value[edge_counter] = alpha * usr_inv_freq[items[k].users[j]];
			edge_counter++;
		}
		//Add row of M_I
		for (sz_long_t j = M_I.csr_row_pointer[k]; j < M_I.csr_row_pointer[k + 1]; j++)
		{
			graph.csr_column[edge_counter] = M_I.csr_column[j] + num_users;
			graph.csr_value[edge_counter] = (1.0 - alpha) * M_I.csr_value[j];
			edge_counter++;
		}
		//Advance row pointer
		graph.csr_row_pointer[i + 1] = edge_counter;
	}

	//free
	for (sz_long_t i = 0; i < M_I.num_nodes; i++)
		free(items[i].users);
	free(items);
	free(item_inv_freq);
	free(usr_inv_freq);
	return graph;
}