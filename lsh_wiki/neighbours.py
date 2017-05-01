import numpy as np
import sframe
from docutils.nodes import inline, copyright

from scipy.sparse import csr_matrix
from scipy.stats._discrete_distns import planck_gen
from sklearn.metrics.pairwise import  paired_distances

from copy import copy
import matplotlib.pyplot as plt
from sympy.physics.quantum.circuitplot import matplotlib

def norm(x):
    sum_sq=x.dot(x.T)
    norm=np.sqrt(sum_sq)
    return norm

#load dataset
wiki=sframe.SFrame('people_wiki.gl/')
wiki=wiki.add_row_number()


# to load tf-idf vectors
def load_sparse_csr(filename):
    loader=np.load(filename)
    data=loader['data']
    indices=loader['indices']
    indptr=loader['indptr']
    shape=loader['shape']

    return csr_matrix((data,indices,indptr),shape)

corpus=load_sparse_csr('people_wiki_tf_idf.npz')

#word to index mapping
map_to_index_to_word=sframe.SFrame('people_wiki_map_index_to_word.gl/')

#Genearte 16 random vectors  of dimension 547979
def generate_random_vectors(num_vector,dim):
    return np.random.randn(dim,num_vector)



np.random.seed(0)
random_vectors=generate_random_vectors(num_vector=16,dim=547979)
print random_vectors.shape

doc=corpus[0,:]

doc=corpus[0,:]
index_bits=(doc.dot(random_vectors) >=0)
power_of_two=(1<<np.arange(15,-1,-1))

print index_bits.dot(power_of_two)

index_bits=corpus.dot(random_vectors)>=0
print  index_bits.dot(power_of_two)

#training
def train_lsh(data,num_vector=16,seed=None):
    dim=data.shape[1]
    if seed is not None:
        np.random.seed(seed)
    random_vectors=generate_random_vectors(num_vector,dim)
    powers_of_two=1<<np.arange(num_vector-1,-1,-1)
    table={}
    bin_index_bits=(data.dot(random_vectors)>=0)
    bin_indices=bin_index_bits.dot(power_of_two)

    for data_index,bin_index in enumerate(bin_indices):
        if bin_index not in table:
            table[bin_index]=[]
        table[bin_index].append(data_index)

    model={'data':data,
           'bin_index_bits':bin_index_bits,
           'bin_indices':bin_indices,
           'table':table,
           'random_vectors':random_vectors,
           'num_vector':num_vector}
    return model

model = train_lsh(corpus, num_vector=16, seed=143)
table = model['table']




doc_ids=list(model['table'][model['bin_indices'][35817]])
doc_ids.remove(35817)
docs=wiki.filter_by(values=doc_ids,column_name='id')
print docs

#compute cosine distance
def cosine_distance(x,y):
    xy=x.dot(y.T)
    dist=xy/(norm(x)*norm(y))
    return 1 - dist[0,0]

obama_tfIdf=corpus[35817,:]
biden_tfIdf=corpus[24478,:]
print "Cosine sitance bwteen Obama"
print "Barack Obama -{0:24s}:{1:f}".format('Joe Biden',cosine_distance(obama_tfIdf,biden_tfIdf))

for doc_id in doc_ids:
    doc_tf_idf=corpus[doc_id,:]
    print "barack Obama -{0:24s}:{1:f}".format(wiki[doc_id]['name'],cosine_distance(obama_tfIdf,doc_tf_idf))


from itertools import combinations
num_vector=16
search_radius=0

for diff in combinations(range(num_vector),search_radius):
    print diff

#SearchNearBy Bins of Radius 2
def search_nearby_bins(query_bin_bits,table,search_radius=2,intial_candidates=set()):
    num_vector=len(query_bin_bits)
    powers_of_two=1<<np.arange(num_vector-1,-1,-1)

    candidate_set=copy(intial_candidates)
    for different_bits in combinations(range(num_vector),search_radius):
        alternate_bits=copy(query_bin_bits)
        for i in different_bits:
            alternate_bits[i]=~alternate_bits[i]
        nearby_bin=alternate_bits.dot(powers_of_two)
        if nearby_bin in table:
            more_docs=table[nearby_bin]
            candidate_set.update(more_docs)
    return candidate_set

obama_bin_index = model['bin_index_bits'][35817] # bin index of Barack Obama
candidate_set = search_nearby_bins(obama_bin_index, model['table'], search_radius=0)
print candidate_set



candidate_set = search_nearby_bins(obama_bin_index, model['table'], search_radius=1,intial_candidates=candidate_set)

from sklearn.metrics import pairwise_distances

#collect all candidates and compute their true distance to query
def query(vec,model,k,max_search_radius):
    data=model['data']
    table=model['table']
    random_vectors=model['random_vectors']
    num_vector=random_vectors.shape[1]

    bin_index_bits=(vec.dot(random_vectors)>=0).flatten()

    #Search Nearby Bins and collect Candidates
    candidate_set=set()

    for search_radius in xrange(max_search_radius+1):
        candidate_set=search_nearby_bins(bin_index_bits,table,search_radius,intial_candidates=candidate_set)
    nearest_neighbours=sframe.SFrame({'id':candidate_set})
    candidates=data[np.array(list(candidate_set)),:]
    nearest_neighbours['distance']=pairwise_distances(candidates,vec,metric='cosine').flatten()
    return nearest_neighbours.topk('distance',k,reverse=True),len(candidate_set)

print query(corpus[35817,:],model,k=10,max_search_radius=3)

result,num_candidates_considered=query(corpus[35817,:],model,k=10,max_search_radius=3)
print result.join(wiki[['id','name']],on='id').sort('distance')


num_candidates_history=[]
query_time_history=[]
max_distance_from_query_history=[]
min_distance_from_query_history=[]
average_distance_from_query_history=[]
import time
for max_search_radius in xrange(17):
    start=time.time()
    result,num_candidates=query(corpus[35817,:],model,k=10,max_search_radius=max_search_radius)
    end=time.time()
    query_time=end-start
    print 'Radius:',max_search_radius
    print result.join(wiki[['id','name']],on='id').sort('distance')
    average_distance_from_query=result['distance'][1:].mean()
    max_distanmce_from_query=result['distance'][1:].max()
    min_distance_from_query=result['distance'][1:].min()

    num_candidates_history.append(num_candidates)
    query_time_history.append(query_time)
    average_distance_from_query_history.append(average_distance_from_query)
    max_distance_from_query_history.append(max_distanmce_from_query)
    min_distance_from_query_history.append(min_distance_from_query)


plt.figure(figsize=(7,4.5))
plt.plot(num_candidates_history,linewidth=4)
plt.xlabel('Search Radius')
plt.ylabel('No of Documents Searched')
plt.rcParams.update({'font.size':16})
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4.5))
plt.plot(query_time_history,linewidth=4)
plt.xlabel('Search Radius')
plt.ylabel('Query time (seconds)')
plt.rcParams.update({'font.size':16})
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4.5))
plt.plot(average_distance_from_query_history, linewidth=4, label='Average of 10 neighbors')
plt.plot(max_distance_from_query_history, linewidth=4, label='Farthest of 10 neighbors')
plt.plot(min_distance_from_query_history, linewidth=4, label='Closest of 10 neighbors')
plt.xlabel('Search radius')
plt.ylabel('Cosine distance of neighbors')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()
plt.show()

# repeat the analysis for entire dataset randomly choosen 10 documents.

def brute_force_query(vec, data, k):
    num_data_points = data.shape[0]

    # Compute distances for ALL data points in training set
    nearest_neighbors = sframe.SFrame({'id': range(num_data_points)})
    nearest_neighbors['distance'] = pairwise_distances(data, vec, metric='cosine').flatten()

    return nearest_neighbors.topk('distance', k, reverse=True)


max_radius = 17
precision = {i: [] for i in xrange(max_radius)}
average_distance = {i: [] for i in xrange(max_radius)}
query_time = {i: [] for i in xrange(max_radius)}

np.random.seed(0)
num_queries = 10
for i, ix in enumerate(np.random.choice(corpus.shape[0], num_queries, replace=False)):
    print('%s / %s' % (i, num_queries))
    ground_truth = set(brute_force_query(corpus[ix, :], corpus, k=25)['id'])
    # Get the set of true nearest neighbors

    for r in xrange(1, max_radius):
        start = time.time()
        result, num_candidates = query(corpus[ix, :], model, k=10, max_search_radius=r)
        end = time.time()

        query_time[r].append(end - start)
        # precision = (# of neighbors both in result and ground_truth)/10.0
        precision[r].append(len(set(result['id']) & ground_truth) / 10.0)
        average_distance[r].append(result['distance'][1:].mean())


plt.figure(figsize=(7,4.5))
plt.plot(range(1,17), [np.mean(average_distance[i]) for i in xrange(1,17)], linewidth=4, label='Average over 10 neighbors')
plt.xlabel('Search radius')
plt.ylabel('Cosine distance')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4.5))
plt.plot(range(1,17), [np.mean(precision[i]) for i in xrange(1,17)], linewidth=4, label='Precison@10')
plt.xlabel('Search radius')
plt.ylabel('Precision')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4.5))
plt.plot(range(1,17), [np.mean(query_time[i]) for i in xrange(1,17)], linewidth=4, label='Query time')
plt.xlabel('Search radius')
plt.ylabel('Query time (seconds)')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size':16})
plt.tight_layout()
plt.show()
