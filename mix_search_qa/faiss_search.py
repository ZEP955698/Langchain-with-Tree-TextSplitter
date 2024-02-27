from config import *

from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore
from langchain.docstore.document import Document
import os
import pandas as pd


class FaissSearch:
    def __init__(self, emb_model_type, vs_path):
        self.emb_model_type = emb_model_type
        self.emb_model_path = EMB_MODEL_DICT[self.emb_model_type]
        self.vs_path = vs_path 

        self.emb_model = None
        self.vector_store = None


    def load(self):
        print("1. 加载Embedding模型中...")
        #### Embedding ####
        if "bge" in self.emb_model_type:
            encode_kwargs = {'normalize_embeddings': True} 
            # model_kwargs = {'device': 'cuda'}
            model_kwargs = {'device': "cuda"} 
            self.emb_model = HuggingFaceBgeEmbeddings(model_name=self.emb_model_path,
                                                      encode_kwargs=encode_kwargs,
                                                      model_kwargs=model_kwargs,
                                                      query_instruction="为这个句子生成表示以用于检索相关文章："
            )
        else:
            raise Exception(f"{self.emb_model_type}暂不支持")
        
        print("2. 加载FAISS库中...")
        #### Faiss #### 
        try:
            if os.path.exists(self.vs_path) and os.path.isdir(self.vs_path):
                self.vector_store = FAISS.load_local(self.vs_path, self.emb_model,)
                # self.vector_store._normalize_L2 = True
            else:
                self.vector_store = FAISS.from_texts(texts=[""], embedding=self.emb_model)
                # self.vector_store._normalize_L2 = True
                self.delete_all()
                # self.vector_store.save_local(self.vs_path)
        except Exception as e:
            raise Exception(e)
        print("3. Embedding模型 & FAISS库 加载完成")


    def get_size(self,):
        return len(self.vector_store.index_to_docstore_id)


    def get_info(self):
        emb_dim = len(self.emb_model.embed_documents(["test"])[0])
        return {"vector_store_path":self.vs_path,
                "vector_store_type":"FAISS",
                "vector_store_size":len(self.vector_store.index_to_docstore_id),
                "emb_model_path":self.emb_model_path,
                "emb_model":self.emb_model_type,
                "emb_dim":emb_dim,
                }


    def add_texts(self, texts, metadatas=None):
        texts = [t.replace("\n", " ") for t in texts]
        device = "cuda" 
        embeddings = self.emb_model.client.encode(texts, 
                                                  batch_size=256,
                                                  show_progress_bar=True,
                                                  normalize_embeddings=True,
                                                  device = device,
                                                  )
        text_embeddings = [(text, embedding) for text, embedding in zip(texts,embeddings)]
        
        if metadatas is None:
            metadatas = ({"source":None} for _ in range(len(texts)))
        else:
            metadatas = (i for i in metadatas)

        self.vector_store.add_embeddings(text_embeddings, metadatas)
        self.vector_store.save_local(self.vs_path)


    def delete_all(self):
        all_ids = [k for k, v in self.vector_store.docstore._dict.items()]
        self.vector_store.delete(all_ids)
        self.vector_store.save_local(self.vs_path)


    def similarity_search(self, query, top_k,):
        result_docs = self.vector_store.similarity_search_with_score(query=query, k=top_k)
        result_texts = [result_docs[i][0].page_content for i in range(len(result_docs))]
        
        scores = [float(result_docs[i][1]) for i in range(len(result_docs))]
        result_metadatas = [result_docs[i][0].metadata for i in range(len(result_docs))]
        # result_metadatas_w_score = [{'score': scores[i], **result_metadatas[i]} for i in range(len(result_metadatas))]
        # results = {result_texts[i]:result_metadatas_w_score[i] for i in range(len(result_docs))}
        
        return result_texts, scores, result_metadatas



