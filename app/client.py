import yaml
import redis
import numpy as np

from openai import OpenAI
from PyPDF2 import PdfReader
from dataclasses import dataclass
from typing import Union, List, Iterable
from redis.commands.search.query import Query
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType


from settings import secret_path

EMBED_MODEL = 'text-embedding-3-small'
@dataclass
class Embedding:
    id: str
    vector: list[float]
    text: str

    def to_dict(self):
        return {
            'id': self.id,
            'vector': self.vector,
            'text': self.text
        }


class OpenAIClient:
    def __init__(self):
        with open(secret_path, 'r') as f:
            __secret = yaml.safe_load(f)
        __api_key = __secret['openai']['api_key']

        if not hasattr(self, 'client') or self.client is None:
            self.client = OpenAI(api_key=__api_key)

        self.model = 'gpt-4o-mini'

    def __del__(self):
        if hasattr(self, 'client') and self.client is not None:
            self.client.close()
            self.client = None

    def chat(self, messages: list[dict]) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return completion.choices[0].message.content
    
    def embeddings(self, input: Union[str, List[str], Iterable[int], Iterable[Iterable[int]]], model: str = EMBED_MODEL):
        return self.client.embeddings.create(
            model=model,
            input=input,
        )
    
    def pdf_to_embeddings(self, pdf_path: str, chunk_size: int = 1000) -> list[Embedding]:
        pdf_reader = PdfReader(pdf_path)
        chunks = []
        for page in pdf_reader.pages:
            text = page.extract_text()
            chunks.extend([text[i:i+chunk_size] for i in range(0, len(text), chunk_size)])

        response = self.embeddings(chunks)

        return [
            Embedding(
                id=value.index,
                vector=value.embedding,
                text=chunks[value.index]
            ) for value in response.data
        ]

class RedisClient:
    def __init__(self):
        with open(secret_path, 'r') as f:
            __secret = yaml.safe_load(f)
        redis_conf = __secret['redis']
        host = redis_conf['host']
        port = redis_conf['port']

        self.client = redis.Redis(host=host, port=port)

    def embeddings_to_redis(self, embeddings: list[Embedding], index_name: str = 'zelda_embeddings'):
        vector_dim = len(embeddings[0].vector)
        vector_num = len(embeddings)

        text = TextField('text')
        text_embedding = VectorField('vector', 
                                     'FLAT', {
                                        'TYPE': 'FLOAT32',
                                        'DIM': vector_dim,
                                        'DISTANCE_METRIC': 'COSINE',
                                        'INITIAL_CAP': vector_num,
                                     })
        fields = [text, text_embedding]

        try:
            self.client.ft(index_name).info()
            print(f"Index {index_name} already exists")
        except:
            print(f"Index {index_name} does not exist, creating...")
            self.client.ft(index_name).create_index(
                fields=fields,
                definition=IndexDefinition(
                    prefix=['embedding'],
                    index_type=IndexType.HASH
                )
            )
            print(f"Index {index_name} created")

        for embedding in embeddings:
            key = f"embedding:{embedding.id}"
            embedding.vector = np.array(embedding.vector, dtype=np.float32).tobytes()
            self.client.hset(key, mapping=embedding.to_dict())

        print(f"Loaded {self.client.info()['db0']['keys']} documents in Redis Search index {index_name}")

    def search_redis(self, 
                     query: str, 
                     index_name: str = 'zelda_embeddings',
                     vector_field: str = 'vector',
                     return_fields: list = ['text', 'score'],
                     k: int = 5,
                     print_results: bool = False,
                     ):
        
        openai_client = OpenAIClient()
        embedding_query = openai_client.embeddings(query).data[0].embedding

        base_query = Query(f'*=>[KNN {k} @{vector_field} $vec_param AS score]')
        query = (
            base_query
            .return_fields(*return_fields)
            .sort_by("score")
            .paging(0, k)
            .dialect(2)
        )

        params = {
            "vec_param": np.array(embedding_query, dtype=np.float32).tobytes()
        }

        results = self.client.ft(index_name).search(query, query_params=params)

        if print_results:
            for i, doc in enumerate(results.docs):
                print(f"{i}. {doc.text} (Score: {round(doc.score, 3)})")

        return [doc.text for doc in results.docs]

