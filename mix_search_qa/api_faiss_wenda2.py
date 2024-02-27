
import uvicorn
import argparse
import pydantic

from io import BytesIO
from typing import Union
from fuzzywuzzy import fuzz
from pydantic import BaseModel

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body, FastAPI, File, Form, Query, UploadFile, WebSocket

from config import *
from utils import *
from faiss_search import FaissSearch
from llm_chat import LLM

from collections import Counter  

import logging
import os
import json
import jieba
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

logging.basicConfig(filename=os.path.join(PROJECT_DIR, "logs/api.log"), 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


#####################################################################
########################## CLASSES  #################################
#####################################################################

class Request(BaseModel):
    query: str = Body(..., description="问题")
    top_k: int = Body(15, description="返回结果数量")
    emb_ratio: float = Body(0.5, description="混合搜索相似度比例")


class SearchResponse(BaseModel):
    code: int = pydantic.Field(200, description="HTTP status code")
    msg: str = pydantic.Field("success", description="HTTP status message")
    # results: list = pydantic.Field(..., description="Results from FAISS")
    fileList: list = pydantic.Field(..., description="Result from FAISS")
    # ids: list = pydantic.Field(..., description="Result ids from FAISS")
    # texts: list = pydantic.Field(..., description="Results from FAISS")
    

class ChatResponse(BaseModel):
    code: int = pydantic.Field(200, description="HTTP status code")
    msg: str = pydantic.Field("success", description="HTTP status message")
    response: str = pydantic.Field(..., description="Response")
    # results: list = pydantic.Field(..., description="Results from FAISS")
    # ids: list = pydantic.Field(..., description="Result ids from FAISS")
    # texts: list = pydantic.Field(..., description="Results from FAISS")
    fileList: list = pydantic.Field(..., description="Result from FAISS")

#####################################################################
########################    UTILS   ################################
#####################################################################

async def cut_list_by_len(the_list, max_len, reverse=False):
    if len(the_list) <= 1:
        return the_list
    if reverse:
        the_list = the_list[::-1]
    result_list = [the_list[0]]
    curr_len = len(the_list[0])
    for i in the_list[1:]:
        if len(i)+curr_len>max_len:
            break
        result_list.append(i)
        curr_len += len(i)
    if reverse:
        result_list = result_list[::-1]
    return result_list


async def depth_first_traverse(results, doc_tree, related_nodes):
    for key, value in doc_tree.items():
        # print(key, value)  # Process the current node
        # results.append(value)
        if key in related_nodes:
            results.append(key)
            related_nodes.remove(key)
        # print(results)
        if isinstance(value, dict):
            # If the value is a dictionary, recursively traverse it
            await depth_first_traverse(results, value, related_nodes)
        # else:
    return results


async def combine_metadatas(metadatas,):
    doc_tree = {}
    related_nodes = set()
    # related_nodes_len = sum([len(i) for i in related_nodes])
    max_len = 200
    for i in range(len(metadatas)):
        metadata = metadatas[i]

        # 前两个匹配的文章片段可以有长一点的上下文长度
        if i>2:
            max_len = 100
        temp = doc_tree

        # 上下文使用长度进行截断
        parents_cut = await cut_list_by_len(metadata["parents"].copy(), max_len, reverse=True)
        children_cut = await cut_list_by_len(metadata["children"].copy(), max_len, reverse=False)
        
        # combine用来构建树，related_node从树中选取匹配的节点
        combined = parents_cut
        combined.append(metadata["content"])
        combined += children_cut
        # combined_len = sum([len(i) for i in combined])

        # if combined_len+related_nodes_len>max_context_len:
        #     break

        related_nodes.update(combined)
        # related_nodes_len = sum([len(i) for i in related_nodes])

        # 构建树
        for j in combined:
            if j not in temp:
                temp[j] = {}
            temp = temp[j]
    # 深度优先搜索树中相关节点
    results = await depth_first_traverse([], doc_tree, related_nodes)
    
    # 去重逻辑，在list中两个片段相比较
    for i in range(len(results)):
        for j in range(len(results)):
            if j<=i:
                continue
            if results[i] is None or results[j] is None:
                continue
            if results[j] in results[i]:
                results[j] = None
            elif results[i] in results[j]:
                results[i] = None
    return [i for i in results if i]


async def faiss_similarity_search(faiss, query, top_k, emb_ratio=0):
    def score2similarity(score):
        return max((2-score)/2, 0)

    vs_size = faiss.get_size()
    # if top_k is None:
    #     top_k = faiss.get_size()
    wide_top_k = top_k * 4 # 防止top_k中有重复答案，所以检索4倍top_k的文本，然后进行过滤
    texts, scores, metadatas = faiss.similarity_search(query, wide_top_k)
    min_score = min(scores)
    max_score = max(scores)
    if len(scores) == 0:
        msg = f"success: 从{vs_size}条数据中, 0条数据成功返回"
        return msg, []
    content_hashes = [calculate_hash(i["content"] + i["content_title"]) for i in metadatas]
    # content_hash_metadata_map_emb = {content_hashes[i]:metadatas[i] for i in range(len(metadatas))}
    content_hash_similarity_map_emb = {content_hashes[i]:(max_score-scores[i])/(max_score-min_score) for i in range(len(metadatas))}

    query_cut = list(jieba.cut(query))
    matched_content_hashes = []
    for kw in query_cut:
        matched_content_hashes += kw_content_hash_map.get(kw, [])
    
    # 建立关键词频率词典，优先返回关键词匹配程度高的
    count_dict = Counter(matched_content_hashes)
    
    # ids = sorted(count_dict, key=lambda x: count_dict[x], reverse=True)
    min_keyword_count = 0
    try:
        max_keyword_count = count_dict.most_common(1)[0][1] # count_dict[id[0]]
        content_hash_similarity_map_kw = {key: (value-min_keyword_count)/max_keyword_count for key, value in count_dict.items()}

    except:
        max_keyword_count = 0
        content_hash_similarity_map_kw = {}
    
    content_hash_similarity_map_all = {content_hash:emb_ratio*content_hash_similarity_map_emb.get(content_hash, 0)+ (1-emb_ratio)*content_hash_similarity_map_kw.get(content_hash, 0) for content_hash in content_hash_similarity_map_emb.keys()}
    sorted_matched_content_hashes = sorted(content_hash_similarity_map_all, key=lambda x: content_hash_similarity_map_all[x], reverse=True)[:top_k]

    metadatas = [content_hash_metadata_map.get(content_hash, None) for content_hash in sorted_matched_content_hashes]
    metadatas = [i for i in metadatas if i]

    # # score列表转为相似度列表
    # similarities = [score2similarity(score) for score in scores]

    # # max_context_len = LLM_MAX_SEQ_LEN - (len(TEMPLATE) + len(query))
    # # candidate_nodes = []
    # # for metadata in metadatas:
    # #     candidate_nodes.append(metadata["content"])

    # # results包含了(匹配的文本，metadata元数据)
    # metadatas = [metadatas[i] for i in range(len(similarities)) if similarities[i]>threshold]

    id_metadatas_map = {}
    id_title_map = {}
    # 构建两个字典，用id作为key，分别检索metadata和title文件名
    for metadata in metadatas:
        if metadata["id"] not in id_metadatas_map:
            id_metadatas_map[metadata["id"]]= []
        id_metadatas_map[metadata["id"]].append(metadata)

        if metadata["id"] not in id_title_map:
            id_title_map[metadata["id"]] = metadata["content_title"]
    
    
    fileList = []
    for k_id, v_metadatas in id_metadatas_map.items():
        combined_v_metadatas = await combine_metadatas(v_metadatas)
        fileList.append({"id":k_id, "title":id_title_map[k_id], "fileSections":combined_v_metadatas})

    # 去重：如果同样的文件有同样的文件内容，但id不同
    prev_file_sections = []
    for i in range(len(fileList)):
        file = fileList[i]
        if file["fileSections"] in prev_file_sections:
            fileList[i] = None
            continue
        prev_file_sections.append(file["fileSections"])
    fileList = [i for i in fileList if i]

    # # 使用fuzzywuzzy进行部分的模糊匹配，如果相似度>85，则删除较短的文本
    # for i in range(len(results)):
    #     for j in range(len(results)):
    #         if i==j or results[j]==None or results[i]==None:
    #             continue
    #         if fuzz.partial_ratio(results[i][0],results[j][0])>90:
    #             if len(results[i][0])>len(results[j][0]):
    #                 results[j] = None
    #             else:
    #                 results[i] = None

    # 选取top_k个结果
    # results = [result for result in results if result][:top_k]

    # num_results = len(results)
    # msg = f"success: 从{vs_size}条数据中, {num_results}条数据成功返回"
    num_results = len(fileList)
    msg = f"success: 从{vs_size}条数据中, {num_results}条数据成功返回"
    return msg, fileList





#####################################################################
#####################    FAISS SEARCH   #############################
#####################################################################


async def similarity_search(search_request: Request):
    query = search_request.query
    top_k = search_request.top_k
    emb_ratio = search_request.emb_ratio
    logger.info(f"api.py: similarity_search(query={query}, top_k={top_k}, emb_ratio={emb_ratio})")
    try:
        msg, fileList = await faiss_similarity_search(tree_baseline_faiss, query, top_k, emb_ratio)
        logger.info(f">>> code=200, msg={msg}, fileList={fileList}\n")
        # result_texts = [result[0] for result in results]
        # result_ids = [result[1]["id"] for result in results ]
        # result_metadatas = [result[1] for result in results]
        # fileList = [{"id":result[1]["id"], "title":result[1]["content_title"], "sections":result[0].split("，")} for result in results]
        
        return SearchResponse(code=200, msg=msg, fileList=fileList)
        # return SearchResponse(code=200, msg=msg, texts=result_texts, ids=result_ids)
    except Exception as e:
        msg = f"fail: {e}"
        logger.info(f">>> code=500, msg={msg}, texts=[], ids=[]\n")
        return SearchResponse(code=500, msg=msg, fileList=[])
        # return SearchResponse(code=500, msg=msg, texts=[], ids=[])



#####################################################################
#######################    LLM CHAT   ###############################
#####################################################################


async def chat(chat_request: Request):
    query = chat_request.query
    top_k = chat_request.top_k
    emb_ratio = chat_request.emb_ratio
    logger.info(f"api.py: chat(query={query}, top_k={top_k}, emb_ratio={emb_ratio})")
    try:
        msg, fileList = await faiss_similarity_search(tree_baseline_faiss, query, top_k, emb_ratio)
        
        # result_ids = [i[1]["id"] for i in results]
        # result_children = [i[1]["children"][0] if len(i[1]["children"])>0 else "" for i in fileList]
        # result_texts = [results[i][0]+result_children[i] for i in range(len(results))]
        result_texts = ["，".join(i["fileSections"]) for i in fileList][::-1]
        result_texts = [i.replace("\\n", "\n").replace("，，", "，").replace("。，", "。").replace("；，", "；").replace("？，", "？") for i in result_texts]
        # result_metadatas = [result[1] for result in results]
        # logger.info(f"\tmsg={msg}, results={results}")
        
        context_max_len = LLM_MAX_SEQ_LEN - len(query) - len(TEMPLATE)
        context = "\n".join(result_texts)[:context_max_len]
        full_query = TEMPLATE.replace("<<context>>",context).replace("<<query>>", query)

        # logger.info(f"\tfull_query={full_query}")

        response, _ = llm_model.chat(full_query, history=[],temperature=0.01)
        # fileList = [{"id":result[1]["id"], "title":result[1]["content_title"], "sections":result[0].split("，")} for result in results]

        logger.info(f">>> code=200, msg={msg}, response={response}, fileList={fileList}\n")
        return ChatResponse(code=200, msg=msg, response=response, fileList=fileList)
    
        # logger.info(f">>> code=200, msg={msg}, response={response}, result_texts={result_texts}, result_ids={result_ids}\n")
        # return ChatResponse(code=200, msg=msg, response=response, texts=result_texts, ids=result_ids)

    except Exception as e:
        msg = f"fail: {e}"
        logger.info(f">>> code=500, msg={msg}, response="", fileList=[]\n")
        return ChatResponse(code=500, msg=msg, response="", fileList=[])


#####################################################################
##########################    END   #################################
#####################################################################


def api_start(host, port):
    global llm_model 
    global tree_baseline_faiss
    global kw_content_hash_map
    global content_hash_metadata_map

    logger.info(f"======== api start ========")

    app = FastAPI()
    
    logger.info(f"======== 1/2. loading llm... ========")
    llm_model = LLM(model_type="chatglm3-6b", ) # Baichuan2-7B-Chat
    llm_model.load(half=True)

    logger.info(f"======== 2/2. loading faiss... ========")
    tree_baseline_faiss = FaissSearch(emb_model_type=DEFAULT_EMB_MODEL, vs_path=os.path.join(PROJECT_DIR, "vector_stores/bge-large-zh/tree_baseline"))
    # tree_baseline_faiss = FaissSearch(emb_model_type="bge-base-zh-v1.5", vs_path=os.path.join(PROJECT_DIR, "vector_stores/bge-base-zh-v1.5/tree_baseline"))
    tree_baseline_faiss.load()

    with open(os.path.join(PROJECT_DIR, "kw_content_hash_map.json"), "r") as json_file1:
        kw_content_hash_map = json.load(json_file1)
    with open(os.path.join(PROJECT_DIR, "content_hash_metadata_map.json"), "r") as json_file2:
        content_hash_metadata_map = json.load(json_file2)

    logger.info(f"======== loading complete! ========")
    # app.post("/faiss/add_from_csv", response_model=BaseResponse)()
    app.post("/faiss/similarity_search", response_model=SearchResponse)(similarity_search)
    app.post("/llm/chat", response_model=ChatResponse)(chat)

    uvicorn.run(app, host=host, port=port)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start chat with chatglm3")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    # parser.add_argument("--device", type=int, default=DEFAULT_LLM_DEVICE)


    # 初始化消息
    args = None
    args = parser.parse_args()
    api_start(args.host, args.port)
    # api_start(args.host, args.port, args.device)
    
