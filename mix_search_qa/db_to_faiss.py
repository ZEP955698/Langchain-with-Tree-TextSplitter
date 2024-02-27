import pymysql
from config import *
from utils import *
from faiss_search import FaissSearch
from tqdm import tqdm
import json
import logging
import jieba

logging.basicConfig(filename=os.path.join(PROJECT_DIR, "logs/db_to_faiss.log"), 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import jieba
# if os.path.exists("mix_search_wenda_dev/jieba_dict.txt"):
jieba.load_userdict("mix_search_wenda_dev/jieba_dict.txt")


#if not os.path.exists(os.path.join(PROJECT_DIR, "doc_tree.jsonl")):
#    os.makedirs(os.path.join(PROJECT_DIR, "doc_tree.jsonl"))
SAVE_to_FAISS = False
SAVE_to_KW_CONTENT_MAP = True
def combine_pre_context(node, max_parent_len=100):
    parents = node["parents"]
    if not parents:
        parents = ""
    else:
        last_parent = parents[-1]
        total_parents_len = sum([len(i) for i in parents])
        while total_parents_len>max_parent_len:
            if parents:
                parents = parents[1:]
                total_parents_len = sum([len(i) for i in parents])
            else:
                break
        if not parents:
            parents = last_parent
        else:
            parents = "，".join(parents)+"，"
    content = node["content"]
    return parents+content



if __name__ == "__main__":
    # try init the FAISS
    if SAVE_to_FAISS:
        logger.info("======== start inital FAISS... ========")
        tree_vs = FaissSearch(emb_model_type="bge-large-zh", vs_path=os.path.join(PROJECT_DIR, "vector_stores/bge-large-zh/tree_baseline"))
        # tree_vs = FaissSearch(emb_model_type="bge-base-zh-v1.5", vs_path=os.path.join(PROJECT_DIR, "vector_stores/bge-base-zh-v1.5/tree_baseline_test"))
        tree_vs.load()
    
    # load keyword set
    logger.info("======== load keyword set... ========")
    kw_path = "mix_search_wenda_dev/jieba_dict.txt"
    with open(kw_path,'r') as f:
        kw = f.readlines()
    kw_set = set([i.strip("\n") for i in set(kw) if len(i.strip("\n"))>1])
    # connect to DB
    logger.info("======== start connect to DB... ========")
    try:
        db_connect_admin = pymysql.connect(host=MYSQL_DICT["host"], port=MYSQL_DICT["port"], 
                                        user=MYSQL_DICT["user"], password=MYSQL_DICT["password"],
                                        database=MYSQL_DICT["database"]) 
                                        #autocommit=MYSQL_DICT["autocommit"], 
                                        #charset=MYSQL_DICT["charset"], connect_timeout=MYSQL_DICT["connect_timeout"])
    except:
        msg = "can't access hte db_connect_admin"
        logger.error(msg)
        raise Exception(msg)


    # process data, and save to FAISS
    content_id_arg = "KL_ID"
    content_title_arg = "KL_CONTENT_TITLE"
    content = "KL_CONTENT"
    logger.info("======== start process data & save to FAISS... ========")
    try:
        sql = f"""SELECT {content_id_arg}, {content_title_arg}, {content} FROM {DB_NAME}"""


        # cursor fetch SQL data
        cursor = db_connect_admin.cursor()
        cursor.execute(sql)
        cursor_data = cursor.fetchall()

        # process data and load to faiss
        texts = []
        metadatas = []
        # content_hash_keywords_map = {}
        keyword_content_hash_map = {}
        content_hash_metadata_map = {}
        for kw in kw_set:
            keyword_content_hash_map[kw] = []
        for i in tqdm(range(len(cursor_data)), desc="processing..."):
            
            content_id, content_title, content = cursor_data[i]

            content_id = content_id if content_id else ""
            content_title = content_title if content_title else ""
            content = content if content else ""
            
            # 从content中转成node列表，每一个node是一个字典（包含parents,content,childrean）
            curr_nodes = process_content(content_id, content_title, content)

            for node in curr_nodes:
                combine = combine_pre_context(node)
                
                texts.append(combine)
                #切分后的片段 hash值保存
                #todo hash
                unique_content_hash = calculate_hash(node["content"] + node["content_title"])
                for combine_kw in kw_set.intersection(set(list(jieba.cut(combine)))):
                    keyword_content_hash_map[combine_kw].append(unique_content_hash)

                content_hash_metadata_map[unique_content_hash] = node
                metadatas.append(node)
            # 批量导入到Faiss，以及保存成用于测试的jsonl文件
            if len(texts)>100000 and SAVE_to_FAISS:
            # if len(texts)>10:
                
                tree_vs.add_texts(texts, metadatas)
                logger.info(f"{len(texts)} of data ({i}/{len(cursor_data)}) has been saved!")
                with open(os.path.join(PROJECT_DIR, "doc_tree.jsonl"), "a+") as jsonl_file:
                    for j in metadatas:
                        jsonl_file.write(json.dumps(j, ensure_ascii=False) + '\n')
                texts = []
                metadatas = []
                
        # 最后一批数据导入
        if len(texts)>0 and SAVE_to_FAISS:
            logger.info(f"{len(texts)} of data has been saved!")
            tree_vs.add_texts(texts, metadatas)

            logger.info("======== successfully loaded to FAISS!!! ========")
        if SAVE_to_KW_CONTENT_MAP:
            with open(os.path.join(PROJECT_DIR, "kw_content_hash_map.json"), "a+") as json_file1:
                json.dump(keyword_content_hash_map, json_file1, ensure_ascii=False,)
            with open(os.path.join(PROJECT_DIR, "content_hash_metadata_map.json"), "a+") as json_file2:
                json.dump(content_hash_metadata_map, json_file2, ensure_ascii=False,)
            logger.info("======== successfully loaded to kw_content_map!!! ========")
    except Exception as e:
        msg = f"can't access to data: {e}"
        logger.error(msg)
        raise Exception(msg)
    finally:
        cursor.close()
        db_connect_admin.close()
        logger.info("======== code finishes ========")
        


