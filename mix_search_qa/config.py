import os 

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMB_MODEL_DICT = {
    # "bge-large-zh-v1.5":"/data/model/embeddings/bge-large-zh-v1.5"
    # "bge-large-zh-v1.5":os.path.join(PROJECT_DIR, "emb_models/bge-large-zh-v1.5")
    "bge-large-zh":"/home/bge-large-zh",
    "bge-base-zh-v1.5":"/home/bge-base-zh-v1.5"
    }

LLM_MODEL_DICT = {
    # "chatglm3-6b":"/data/model/chatglm3-6b",
    "chatglm3-6b":"/home/chatglm3-6b",
    "Qwen-7B":"/data/QwenQwen-7B",
    "Qwen-7B-Chat":"/data/QwenQwen-7B-Chat",
    "Baichuan2-7B-Chat":"/data/Baichuan2-7B-Chat",
}

DEFAULT_EMB_MODEL = "bge-large-zh" 
DEFAULT_LLM_MODEL = "chatglm3-6b"


DEFAULT_EMB_MODEL_PATH = EMB_MODEL_DICT[DEFAULT_EMB_MODEL]
DEFAULT_LLM_MODEL_PATH = LLM_MODEL_DICT[DEFAULT_LLM_MODEL]

SUPPORTED_LLM_MODELS = LLM_MODEL_DICT.keys()
SUPPORTED_EMB_MODELS = EMB_MODEL_DICT.keys()

# 注意context上下文长度，不要超过大模型限制；切换LLM是记得修改LLM_MAX_SEQ_LEN
TEMPLATE = """<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。</指令>\n <已知信息><<context>></已知信息>、\n<问题><<query>></问题>"""
LLM_MAX_SEQ_LEN = 8192



DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 7995
# DEFAULT_PORT = 7998
# DEFAULT_LLM_DEVICE = 7
# DEFAULT_EMB_DEVICE = 5


MYSQL_DICT = {
    "host":"XXX.XXX.XXX.XXX",
    "port":100,
    "user":"XXX",
    "password":"XXX",
    "database":"XXX",
    "autocommit":True,
    "charset":"utf8mb4",
    "chaconnect_timeoutrset":100,
}

DB_NAME = "DB"
