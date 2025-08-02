# services/vector_db.py

import openai
import chromadb
import tenacity
import logging
import hashlib
import time

# 从重构后的模块中导入依赖
from config import Config

class EmbeddingModel:
    """
    封装与嵌入模型 API 的交互。
    负责将文本批量转换为向量表示。
    """
    def __init__(self, config: Config):
        self.config = config
        self.client = None
        self.model_name = config.embedding_model_name

        if not config.embedding_api_base_url or not config.embedding_api_key:
            logging.warning("EMBEDDING_API_BASE_URL 或 EMBEDDING_API_KEY 未设置。嵌入功能将被禁用。")
            return

        try:
            # 假设嵌入服务的 API 遵循 OpenAI 的接口规范
            self.client = openai.OpenAI(
                base_url=config.embedding_api_base_url,
                api_key=config.embedding_api_key,
                timeout=300.0
            )
            logging.info(f"嵌入客户端初始化成功，连接至 {config.embedding_api_base_url}，模型为 '{self.model_name}'")
        except Exception as e:
            logging.error(f"初始化嵌入客户端时出错: {e}")
            self.client = None

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        为文本列表批量获取嵌入向量，内置健壮的重试机制。
        """
        if not self.client:
            logging.error("嵌入客户端不可用。无法获取嵌入向量。")
            return [[] for _ in texts]

        all_embeddings = []
        for i in range(0, len(texts), self.config.embedding_batch_size):
            batch_texts = texts[i:i + self.config.embedding_batch_size]
            
            try:
                retryer = tenacity.Retrying(
                    wait=tenacity.wait_exponential(
                        multiplier=self.config.api_retry_wait_multiplier,
                        min=2,
                        max=self.config.api_retry_max_wait
                    ),
                    stop=tenacity.stop_after_attempt(self.config.api_retry_max_attempts),
                    retry=(
                        tenacity.retry_if_exception_type(openai.APITimeoutError) |
                        tenacity.retry_if_exception_type(openai.APIConnectionError) |
                        tenacity.retry_if_exception_type(openai.InternalServerError) |
                        tenacity.retry_if_exception_type(openai.RateLimitError)
                    ),
                    before_sleep=tenacity.before_sleep_log(logging.getLogger(__name__), logging.WARNING),
                    reraise=True
                )
                response = retryer(
                    self.client.embeddings.create,
                    model=self.model_name,
                    input=batch_texts
                )
                embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings)
                logging.info(f"  成功为批次 {i // self.config.embedding_batch_size + 1} 获取 {len(embeddings)} 个嵌入向量。")
            except Exception as e_retry:
                logging.error(f"  在重试后未能为批次 {i // self.config.embedding_batch_size + 1} 获取嵌入向量: {e_retry}", exc_info=True)
                all_embeddings.extend([[] for _ in batch_texts]) # 为失败的批次添加空列表
        return all_embeddings

    def get_embedding(self, text: str) -> list[float]:
        """获取单个文本的嵌入向量。"""
        embeddings = self.get_embeddings([text])
        return embeddings[0] if embeddings and embeddings[0] else []


class VectorDBManager:
    """
    封装与 ChromaDB 向量数据库的交互。
    负责数据的持久化存储和相似性检索。
    """
    def __init__(self, config: Config, embedding_model: EmbeddingModel):
        self.config = config
        self.embedding_model = embedding_model
        self.client = None
        self.collection = None
        self._initialize_db()

    def _initialize_db(self):
        """初始化数据库客户端和集合（Collection）。"""
        try:
            logging.info(f"正在初始化 ChromaDB，路径: {self.config.vector_db_path}")
            # 使用持久化客户端，确保数据在程序关闭后依然存在
            self.client = chromadb.PersistentClient(path=self.config.vector_db_path)
            self.collection = self.client.get_or_create_collection(
                name=self.config.vector_db_collection_name
            )
            logging.info(f"ChromaDB 集合 '{self.config.vector_db_collection_name}' 已加载/创建。")
            self.get_db_stats()
        except Exception as e:
            logging.error(f"初始化 ChromaDB 失败: {e}", exc_info=True)
            self.client = None
            self.collection = None

    def add_experience(self, texts: list[str], metadatas: list[dict] | None = None, ids: list[str] | None = None) -> bool:
        """向数据库中添加新的经验（文本、元数据等）。"""
        if not self.collection or not self.embedding_model:
            logging.error("向量数据库或嵌入模型未初始化。无法添加经验。")
            return False
        if not texts:
            logging.warning("试图向经验库添加一个空文本列表。")
            return False

        try:
            embeddings = self.embedding_model.get_embeddings(texts)
            
            valid_texts, valid_embeddings, valid_metadatas, valid_ids = [], [], [], []

            for i, emb in enumerate(embeddings):
                if emb: # 仅处理成功生成嵌入向量的文本
                    valid_texts.append(texts[i])
                    valid_embeddings.append(emb)
                    if metadatas: valid_metadatas.append(metadatas[i])
                    # 如果没有提供 ID，则基于内容和时间戳生成一个
                    if ids: 
                        valid_ids.append(ids[i])
                    else: 
                        valid_ids.append(f"exp_{hashlib.md5(texts[i].encode()).hexdigest()}_{int(time.time())}")
                else:
                    logging.warning(f"未能为文本生成嵌入向量，跳过添加: {texts[i][:100]}...")

            if not valid_texts:
                logging.warning("未生成有效的嵌入向量。不会添加新的经验。")
                return False

            logging.info(f"正在向向量数据库添加 {len(valid_texts)} 条经验...")
            self.collection.add(
                embeddings=valid_embeddings,
                documents=valid_texts,
                metadatas=valid_metadatas if valid_metadatas else None,
                ids=valid_ids
            )
            logging.info(f"成功向数据库添加 {len(valid_texts)} 条经验。")
            self.get_db_stats()
            return True
        except Exception as e:
            logging.error(f"向向量数据库添加经验失败: {e}", exc_info=True)
            return False

    def retrieve_experience(self, query_text: str, n_results: int = -1, where_filter: dict | None = None) -> list[dict]:
        """根据查询文本，从数据库中检索最相似的经验。"""
        if not self.collection or not self.embedding_model:
            logging.error("向量数据库或嵌入模型未初始化。无法检索经验。")
            return []
        
        if n_results == -1: 
            n_results = self.config.num_retrieved_experiences

        try:
            logging.info(f"正在从向量数据库检索与查询相关的经验: '{query_text[:100]}...' (前 {n_results} 条)")
            query_embedding = self.embedding_model.get_embedding(query_text)
            if not query_embedding:
                logging.error("未能为查询文本生成嵌入向量。无法检索。")
                return []

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                include=['documents', 'metadatas', 'distances']
            )

            retrieved_experiences = []
            if results and results.get('ids') and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    exp = {
                        "id": results['ids'][0][i],
                        "document": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i],
                    }
                    retrieved_experiences.append(exp)
                logging.info(f"成功检索到 {len(retrieved_experiences)} 条经验。")
            else:
                logging.info("未找到相关的经验。")
            return retrieved_experiences
        except Exception as e:
            logging.error(f"从向量数据库检索经验失败: {e}", exc_info=True)
            return []

    def get_db_stats(self):
        """获取并打印数据库的统计信息。"""
        if not self.collection:
            logging.info("向量数据库集合未初始化。无法获取统计信息。")
            return {}
        try:
            count = self.collection.count()
            logging.info(f"向量数据库 '{self.config.vector_db_collection_name}' 当前包含 {count} 条经验。")
            return {"count": count}
        except Exception as e:
            logging.error(f"获取向量数据库统计信息失败: {e}", exc_info=True)
            return {}