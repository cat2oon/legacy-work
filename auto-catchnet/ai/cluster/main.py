import faiss
import numpy as np

from ac.common.randoms import RandomSeed


"""
 Parameters
"""
RandomSeed().set_numpy_seed()

num_neighbors = 400             # 클러스터 개수 (look vec 히스토그램으로 확인해 볼 것)
num_dimension = 515             # model vector 크기
num_query = 10000               # 쿼리 아이템 개수
num_database = 8000000          # unity 1681 class 전체

res = faiss.StandardGpuResources()

"""
 데이터 준비
"""
# database = None               # database.shape = (num_database, num_dimension)
database = np.random.random((num_database, num_dimension)).astype(np.float32)
database[:, 0] += np.arange(num_database) / 1000.

query = np.random.random((num_query, num_dimension)).astype(np.float32)
query[:, 0] += np.arange(num_query) / 1000.

"""
 Index 구성  
"""
index = faiss.IndexFlatL2(num_dimension)
index.add(database)
print("index total num: {}".format(index.ntotal))

"""
 pre-search
"""
distances, indexes = index.search(database[:5], num_neighbors)
print(zip(indexes, distances))

"""
 Search
"""
distances, indexes = index.search(query, num_neighbors)







