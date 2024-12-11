import faiss
import sqlite3

class FAISSDatabase:
    def __init__(self, index_path, dimension, metadata_path="metadata.db"):
        self.index = faiss.IndexFlatL2(dimension)
        self.index_path = index_path
        self.conn = sqlite3.connect(metadata_path)
        self._create_metadata_table()

    def _create_metadata_table(self):
        with self.conn:
            self.conn.execute('''CREATE TABLE IF NOT EXISTS metadata 
                                 (vector_id TEXT, image_id TEXT)''')

    def add_vector(self, vector_id, image_id, vector):
        self.index.add(vector)
        with self.conn:
            self.conn.execute("INSERT INTO metadata VALUES (?, ?)", (vector_id, image_id))

    def save(self):
        faiss.write_index(self.index, self.index_path)
