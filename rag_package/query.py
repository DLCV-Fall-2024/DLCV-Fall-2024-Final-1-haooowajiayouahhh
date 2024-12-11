import faiss
import sqlite3

class RAGQuery:
    def __init__(self, index_path, metadata_path):
        self.index = faiss.read_index(index_path)
        self.conn = sqlite3.connect(metadata_path)

    def search(self, image_id, k=5):
        query_vector_id = (self.conn.execute("SELECT vector_id FROM metadata WHERE image_id=?", (image_id,)).fetchone())
        if query_vector_id is None:
            raise ValueError(f"Image ID {image_id} not found in metadata!")
        query_vector_id = query_vector_id[0]
        # print(f"query_vector_id:{query_vector_id}")
        # print(f"type(query_vector_id):{type(query_vector_id)}")
        query_vector_id = int(query_vector_id)
        query_vector = self.index.reconstruct(query_vector_id)
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Get k nearest neighbors
        distances, indices = self.index.search(query_vector, k)
        image_ids = []
        print(f"vector_id:{indices}")
        for idx in indices[0]:
            image_ids.append(self.conn.execute("SELECT image_id FROM metadata WHERE vector_id=?", (str(idx),)).fetchone()[0])

        return distances, indices, image_ids
