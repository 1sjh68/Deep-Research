# services/web_research.py (最终修正版)

import os
import logging
import re
import json
import socket
import asyncio
import ssl
import certifi

import aiohttp
import httplib2
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import fitz  # PyMuPDF

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_httplib2 import AuthorizedHttp
from google.oauth2.service_account import Credentials

# 从重构后的模块中导入依赖
from config import Config
from services.llm_interaction import call_ai
from utils.text_processor import truncate_text_for_context


def get_google_auth_http(config: Config):
    proxy_url = os.environ.get("HTTP_PROXY")
    proxy_info = None
    if proxy_url:
        parsed_proxy = urlparse(proxy_url)
        proxy_info = httplib2.ProxyInfo(
            proxy_type=httplib2.socks.PROXY_TYPE_HTTP,
            proxy_host=parsed_proxy.hostname,
            proxy_port=parsed_proxy.port,
            proxy_user=parsed_proxy.username,
            proxy_pass=parsed_proxy.password
        )
        logging.info(f"Google API 将使用代理: {parsed_proxy.hostname}:{parsed_proxy.port}")

    if config.google_service_account_path and os.path.exists(config.google_service_account_path):
        try:
            creds = Credentials.from_service_account_file(
                config.google_service_account_path, 
                scopes=['https://www.googleapis.com/auth/cse']
            )
            return AuthorizedHttp(creds, http=httplib2.Http(proxy_info=proxy_info, timeout=config.api_request_timeout_seconds))
        except Exception as e:
            logging.error(f"未能使用服务账户进行 Google API 认证，回退到默认方式。错误: {e}")
    
    return httplib2.Http(proxy_info=proxy_info, timeout=config.api_request_timeout_seconds)

def perform_search(config: Config, query: str) -> list[dict]:
    if not config.google_api_key or not config.google_cse_id:
        logging.error("Google API 密钥或 CSE ID 未配置。搜索功能已禁用。")
        return []
    try:
        logging.info(f"正在执行 Google 搜索: '{query}'")
        http_auth = get_google_auth_http(config)
        service = build("customsearch", "v1", developerKey=config.google_api_key, http=http_auth)
        res = service.cse().list(q=query, cx=config.google_cse_id, num=config.num_search_results).execute()
        items = res.get('items', [])
        logging.info(f"  Google 搜索为 '{query}' 返回了 {len(items)} 个结果。")
        return items
    except Exception as e:
        logging.error(f"  Google 搜索期间发生未知错误: {e}", exc_info=True)
        return []

def create_intelligent_search_queries(config: Config, knowledge_gap: str, full_document_context: str) -> list[str]:
    logging.info(f"  正在为知识空白生成智能搜索查询: '{knowledge_gap[:100]}...'")
    context_summary_for_query_gen = truncate_text_for_context(config, full_document_context, 1000, "middle")
    prompt = f"""
    根据以下提供的“知识空白”和“文档上下文摘要”，生成1-3个高度具体且有效的搜索引擎查询。
    查询应简洁，并使用关键词。

    知识空白: "{knowledge_gap}"
    文档上下文摘要: --- {context_summary_for_query_gen} ---
    生成的搜索查询 (每行一个，最多3个):
    """
    messages = [{"role": "user", "content": prompt}]
    response_content = call_ai(config, config.researcher_model_name, messages, temperature=0.2, max_tokens_output=150)
    if "AI模型调用失败" in response_content or not response_content.strip():
        return [knowledge_gap]
    queries = [q.strip() for q in response_content.splitlines() if q.strip()]
    if not queries: return [knowledge_gap]
    logging.info(f"  为知识空白“{knowledge_gap[:50]}...”生成了 {len(queries)} 个查询: {queries}")
    return queries[:config.max_queries_per_gap]

async def scrape_and_summarize_async(session: aiohttp.ClientSession, config: Config, url: str, knowledge_gap: str, specific_query: str) -> str:
    logging.info(f"  [ASYNC] 抓取和总结中: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        proxy_str = os.environ.get('HTTP_PROXY')
        ssl_context = ssl.create_default_context(cafile=certifi.where())

        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30), proxy=proxy_str, ssl=ssl_context) as response:
            if response.status != 200:
                logging.warning(f"  [ASYNC] 抓取 {url} 失败，状态码：{response.status}。")
                return ""
            
            text_content = ""
            if 'application/pdf' in response.headers.get('Content-Type', '').lower():
                pdf_bytes = await response.read()
                with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                    text_content = "".join(page.get_text() for page in doc)
            elif 'text/html' in response.headers.get('Content-Type', '').lower():
                html_content = await response.text()
                soup = BeautifulSoup(html_content, 'lxml')
                for script_or_style in soup(["script", "style"]):
                    script_or_style.decompose()
                text_content = soup.get_text(separator='\n', strip=True)
            
            if not text_content.strip(): return ""

            summary_prompt = f"""请根据以下针对知识空白“{knowledge_gap}”的特定查询“{specific_query}”来总结下面的文本。提取与此查询最相关的信息。
            文本内容：--- {truncate_text_for_context(config, text_content, 6000)} ---
            总结：
            """
            
            # --- 关键修正点 2 ---
            # 在异步函数中调用同步的 call_ai 函数，必须使用 asyncio.to_thread
            summary = await asyncio.to_thread(
                call_ai, config, config.summary_model_name, [{"role": "user", "content": summary_prompt}], max_tokens_output=768
            )
            
            if "AI模型调用失败" not in summary and summary.strip():
                return f"URL: {url}\n查询: {specific_query}\n总结:\n{summary}\n"
            return ""
    except Exception as e:
        logging.error(f"  [ASYNC] 抓取和总结 {url} 时发生未知错误: {e}", exc_info=False) # 减少日志冗余
        return ""

async def run_research_cycle_async(config: Config, knowledge_gaps: list[str], full_document_context: str) -> str:
    if not knowledge_gaps: return ""
    logging.info(f"\n--- 开始为 {len(knowledge_gaps)} 个知识空白进行智能知识发现 ---")
    all_tasks = []
    final_brief_list = []

    async with aiohttp.ClientSession() as session:
        for gap_text in knowledge_gaps:
            search_queries = await asyncio.to_thread(create_intelligent_search_queries, config, gap_text, full_document_context)
            for query in search_queries:
                search_results = await asyncio.to_thread(perform_search, config, query)
                for res_item in search_results:
                    if url := res_item.get('link'):
                        task = scrape_and_summarize_async(session, config, url, gap_text, query)
                        all_tasks.append(task)
        
        # --- 关键修正点 1 ---
        # 任务的执行和结果处理，必须在 session 保持打开状态的 `async with` 块内完成
        if not all_tasks:
            logging.warning("--- 异步研究周期中未创建任何有效的抓取任务 ---")
            return ""

        logging.info(f"正在并发执行 {len(all_tasks)} 个抓取/摘要任务...")
        completed_briefs = await asyncio.gather(*all_tasks)
        final_brief_list = [str(b) for b in completed_briefs if b and isinstance(b, str)]

    if not final_brief_list:
        logging.warning("--- 智能研究周期未产生任何有效简报 ---")
        return ""
        
    logging.info(f"--- 知识发现完成，生成了 {len(final_brief_list)} 份简报 ---")
    return "\n\n===== 研究简报开始 =====\n\n" + "\n".join(final_brief_list) + "\n===== 研究简报结束 =====\n\n"