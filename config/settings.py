# config/settings.py

import os
import sys
import logging
from datetime import datetime
import tiktoken

# 尝试从 .env 文件加载环境变量
try:
    from dotenv import load_dotenv
    # 尽早初始化日志，以捕获加载过程的消息
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    logging.info("正在尝试加载 .env 文件...")
    if load_dotenv():
        logging.info(".env 文件加载成功。")
    else:
        logging.info(".env 文件未找到或为空，将依赖系统环境变量。")
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    logging.info("dotenv 库不可用，将依赖系统环境变量。")
except Exception as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    logging.warning(f"加载 .env 文件时出错: {e}")

class Config:
    def __init__(self):
        # 主要 LLM API 配置 (DeepSeek)
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

        # 嵌入模型 API 配置
        self.embedding_api_base_url = os.getenv("EMBEDDING_API_BASE_URL")
        self.embedding_api_key = os.getenv("EMBEDDING_API_KEY")
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "bge-m3")

        # Google 搜索 API 配置
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_cse_id = os.getenv("GOOGLE_CSE_ID")
        self.google_service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        # --- 向量数据库配置 ---
        self.vector_db_path = os.getenv("VECTOR_DB_PATH", "./chroma_db")
        self.vector_db_collection_name = os.getenv("VECTOR_DB_COLLECTION_NAME", "experience_store")
        self.embedding_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", 32))
        self.num_retrieved_experiences = int(os.getenv("NUM_RETRIEVED_EXPERIENCES", 3))

        # 模型名称
        self.main_ai_model = os.getenv("MAIN_AI_MODEL", "deepseek-chat")
        self.main_ai_model_heavy = os.getenv("MAIN_AI_MODEL_HEAVY", "deepseek-reasoner")
        self.secondary_ai_model = os.getenv("SECONDARY_AI_MODEL", "deepseek-reasoner")
        self.summary_model_name = os.getenv("SUMMARY_MODEL_NAME", "deepseek-coder")
        self.researcher_model_name = os.getenv("RESEARCHER_MODEL_NAME", "deepseek-reasoner")
        self.outline_model_name = os.getenv("OUTLINE_MODEL_NAME", "deepseek-coder")
        self.planning_review_model_name = os.getenv("PLANNING_REVIEW_MODEL_NAME", "deepseek-coder")
        self.editorial_model_name = os.getenv("EDITORIAL_MODEL_NAME", self.main_ai_model)
        self.json_fixer_model_name = os.getenv("JSON_FIXER_MODEL_NAME", "deepseek-coder")

        # 分词器
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            print(f"严重错误：初始化 tiktoken 编码器失败: {e}", file=sys.stderr)
            self.encoder = None

        # 路径和目录
        self.checkpoint_file_name = "optimization_checkpoint_outline.json"
        self.session_base_dir = "output"
        self.session_dir = "" 
        self.log_file_path = ""

        # 数值参数
        self.api_request_timeout_seconds = int(os.getenv("API_TIMEOUT_SECONDS", 900))
        self.max_iterations = int(os.getenv("MAX_ITERATIONS", 7))
        self.initial_solution_target_chars = int(os.getenv("INITIAL_SOLUTION_TARGET_CHARS", 20000))
        self.max_context_for_long_text_review_tokens = int(os.getenv("MAX_CONTEXT_TOKENS_REVIEW", 30000))
        self.intermediate_edit_max_tokens = int(os.getenv("INTERMEDIATE_EDIT_MAX_TOKENS", 8192))
        
        # API 调用重试参数
        self.api_retry_max_attempts = int(os.getenv("API_RETRY_MAX_ATTEMPTS", 3))
        self.api_retry_wait_multiplier = int(os.getenv("API_RETRY_WAIT_MULTIPLIER", 1))
        self.api_retry_max_wait = int(os.getenv("API_RETRY_MAX_WAIT", 60))

        self.max_chunk_tokens = int(os.getenv("MAX_CHUNK_TOKENS", 4096))
        self.overlap_chars = int(os.getenv("OVERLAP_CHARS", 800))
        self.max_chunks_per_section = int(os.getenv("MAX_CHUNKS_PER_SECTION", 20))
        self.num_search_results = int(os.getenv("NUM_SEARCH_RESULTS", 3))
        self.max_queries_per_gap = int(os.getenv("MAX_QUERIES_PER_GAP", 5))
        self.min_chars_for_short_chunk_warning = int(os.getenv("MIN_CHARS_SHORT_CHUNK", 50))
        self.min_allocated_chars_for_section = int(os.getenv("MIN_ALLOCATED_CHARS_SECTION", 100))

        # 布尔标志
        self.interactive_mode = os.getenv("INTERACTIVE_MODE", "False").lower() == "true"
        self.use_async_research = os.getenv("USE_ASYNC_RESEARCH", "True").lower() == "true"
        self.enable_dynamic_outline_correction = os.getenv("ENABLE_DYNAMIC_OUTLINE_CORRECTION", "True").lower() == "true"

        # 用户特定输入
        self.user_problem = ""
        self.external_data_files = []

        self.prompts = {}
        self.client = None # 将在初始化后被赋值为 openai.OpenAI 实例

    def setup_logging(self, logging_level=logging.INFO):
        now = datetime.now()
        session_timestamp = now.strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.session_base_dir, f"session_{session_timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)
        self.log_file_path = os.path.join(self.session_dir, "session.log")

        root_logger = logging.getLogger()
        if root_logger.hasHandlers():
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
                handler.close()

        logging.basicConfig(
            level=logging_level,
            format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file_path, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info(f"日志记录已初始化。会话目录: {self.session_dir}")
        logging.info(f"日志文件: {self.log_file_path}")

    def _initialize_deepseek_client(self):
        # 这个方法依赖 openai，应该在 main.py 或 services 模块中调用
        # 这里只保留定义
        import openai # 延迟导入
        if not self.deepseek_api_key:
            logging.critical("DEEPSEEK_API_KEY 环境变量未设置。")
            raise ValueError("DEEPSEEK_API_KEY 环境变量未设置。")
        try:
            self.client = openai.OpenAI(
                api_key=self.deepseek_api_key,
                base_url=self.deepseek_base_url,
                timeout=float(self.api_request_timeout_seconds)
            )
            logging.info(f"DeepSeek 客户端初始化成功，连接至 {self.deepseek_base_url}")
        except Exception as e:
            logging.critical(f"DeepSeek 客户端初始化期间出错: {e}。请检查 API 密钥/URL 和网络连接。")
            raise RuntimeError(f"DeepSeek 客户端初始化失败: {e}")

    def count_tokens(self, text: str) -> int:
        if not text: return 0
        if self.encoder:
            try:
                return len(self.encoder.encode(text))
            except Exception as e:
                logging.warning(f"Tiktoken 编码失败: {e}。回退到近似计算。")
                return len(text) // 4
        logging.warning("Tiktoken 编码器不可用，Token 计数使用近似值 (长度 // 4)。")
        return len(text) // 4# -*- coding: utf-8 -*-

