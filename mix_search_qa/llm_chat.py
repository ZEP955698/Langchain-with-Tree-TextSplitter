import torch

from transformers import AutoTokenizer
from transformers import AutoModel, AutoModelForCausalLM
# from transformers.generation import GenerationConfig
from transformers.generation.utils import GenerationConfig


from config import *

class LLM:
    def __init__(self, model_type):
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        

    def load(self, half=True):
        if self.model_type in SUPPORTED_LLM_MODELS:
            model_path = LLM_MODEL_DICT[self.model_type]  
        # ============ Load tokenizer ============
        if self.model_type in ["chatglm3-6b", "Qwen-7B", "Qwen-7B-Chat"]:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        elif self.model_type in ["Baichuan2-7B-Chat"]:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        else:
            raise Exception(f"{self.model_type} not supported")
        
        # ============ Load model ============
        if self.model_type in ["chatglm3-6b"]:
            self.model = AutoModel.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
        if self.model_type in ["Qwen-7B", "Qwen-7B-Chat"]:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
        if self.model_type in ["Baichuan2-7B-Chat"]:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
        
        # ============ eval model ============
        if self.model_type in ["chatglm3-6b", "Qwen-7B", "Qwen-7B-Chat"]:
            self.model = self.model.half() if half else self.model
            # self.model = self.model.to(f'cuda:{device}') if device is not None else self.model
            self.model = self.model.eval()
        elif self.model_type in ["Baichuan2-7B-Chat"]:
            self.model.generation_config = GenerationConfig.from_pretrained(model_path)

        # ============ not valid model ============
        else:
            raise Exception(f"{self.model_type} not supported")
        


    def chat(self, query, history=[], temperature=0.01):
        
        if self.model_type in ["chatglm3-6b", "Qwen-7B-Chat"]:
            response, history = self.model.chat(self.tokenizer, query, history=history, temperature=temperature)
        elif self.model_type in ["Qwen-7B"]:
            inputs = self.tokenizer(query, return_tensors='pt').to(self.model.device)
            pred = self.model.generate(**inputs)
            response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
            history = []
        elif self.model_type in ["Baichuan2-7B-Chat"]:
            messages = [{"role": "user", "content": query}]
            response = self.model.chat(self.tokenizer, messages)
            history = []
        else:
            raise Exception(f"{self.model_type} not supported")
        
        return response, history
