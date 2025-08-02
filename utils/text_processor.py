# utils/text_processor.py

import logging
import re
import hashlib
import json

# 从重构后的模块中导入依赖
from config import Config
from services.llm_interaction import call_ai

def truncate_text_for_context(config: Config, text: str, max_tokens: int, truncation_style: str = "middle") -> str:
    """
    根据 token 数量安全地截断文本，以适应模型的上下文窗口。
    """
    if not text: 
        return ""
        
    # 确保 encoder 已经初始化
    if not config.encoder:
        logging.warning("Tiktoken 编码器不可用，将使用基于字符的近似截断。")
        # 提供一个基于字符数的备用截断方案
        char_limit = max_tokens * 3 # 假设一个 token 约等于3个字符
        if len(text) <= char_limit:
            return text
        logging.info(f"    - 正在截断文本: {len(text)} chars -> {char_limit} chars (方式: {truncation_style})")
        if truncation_style == "head": return text[:char_limit] + "\n... [内容已截断] ..."
        if truncation_style == "tail": return "... [内容已截断] ...\n" + text[-char_limit:]
        half = char_limit // 2
        return text[:half] + "\n... [中间内容已截断] ...\n" + text[-half:]

    tokens = config.encoder.encode(text)
    if len(tokens) <= max_tokens: 
        return text
    
    logging.info(f"    - 正在截断文本: {len(tokens)} tokens -> {max_tokens} tokens (方式: {truncation_style})")
    
    decode_fn = config.encoder.decode
    if truncation_style == "head":
        truncated_tokens = tokens[:max_tokens]
        return decode_fn(truncated_tokens) + "\n... [内容已截断，只显示开头部分] ..."
    elif truncation_style == "tail":
        truncated_tokens = tokens[-max_tokens:]
        return "... [内容已截断，只显示结尾部分] ...\n" + decode_fn(truncated_tokens)
    else:  # middle
        h_len = max_tokens // 2
        t_len = max_tokens - h_len
        head_part = decode_fn(tokens[:h_len])
        tail_part = decode_fn(tokens[-t_len:])
        return head_part + "\n... [中间内容已截断] ...\n" + tail_part

def calculate_checksum(data: str) -> str:
    """计算字符串的 SHA256 校验和，用于比较内容是否有变化。"""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()

def preprocess_json_string(json_string: str) -> str:
    """
    应用一系列正则表达式修复常见的 LLM 生成的 JSON 错误。
    """
    if not json_string or json_string.isspace():
        return ""

    processed_string = json_string.strip()
    
    # 移除 JS/C++ 风格的注释 (// 和 /* */)
    processed_string = re.sub(r"//.*", "", processed_string)
    processed_string = re.sub(r"/\*[\s\S]*?\*/", "", processed_string, flags=re.MULTILINE)

    # 提取被 markdown 代码块包裹的 JSON
    match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', processed_string, re.DOTALL)
    if match:
        processed_string = match.group(1).strip()
    
    # 移除结尾的逗号 (trailing commas)
    processed_string = re.sub(r",\s*([}\]])", r"\1", processed_string)
    
    # 替换 Python/JS 的布尔值和 None/null
    processed_string = re.sub(r"\bTrue\b", "true", processed_string)
    processed_string = re.sub(r"\bFalse\b", "false", processed_string)
    processed_string = re.sub(r"\bNone\b", "null", processed_string)
    
    return processed_string


def _extract_json_from_ai_response(config: Config, response_text: str, context_for_error_log: str = "AI response") -> str | None:
    """
    使用“三振出局”策略从 AI 的文本响应中稳健地提取 JSON 字符串。
    策略: 1. 直接解析 -> 2. 正则表达式预处理后解析 -> 3. 调用 AI 修复后解析
    """
    logging.debug(f"尝试从以下内容提取JSON: {response_text[:300]}... 上下文: {context_for_error_log}")

    def _try_parse(s_to_parse, stage_msg):
        """尝试解析字符串，如果成功返回字符串，否则返回 None。"""
        if not s_to_parse or s_to_parse.isspace():
            return None
        try:
            json.loads(s_to_parse)
            logging.info(f"  JSON 在 {stage_msg} 阶段解析成功。")
            return s_to_parse
        except json.JSONDecodeError:
            logging.debug(f"  JSON 在 {stage_msg} 阶段解析失败。")
            return None

    # 尝试1: 直接解析原始响应
    if (parsed_str := _try_parse(response_text, "直接解析")) is not None:
        return parsed_str

    # 尝试2: 预处理后解析
    pre_repaired_str = preprocess_json_string(response_text)
    if (parsed_str := _try_parse(pre_repaired_str, "正则预处理")) is not None:
        return parsed_str

    # 尝试3: 调用 AI进行修复 (最后的手段)
    logging.info(f"  JSON 解析在预处理后仍然失败，尝试调用 AI 修复...")
    fixer_prompt = (
        "The following text is supposed to be a valid JSON string, but it's malformed. "
        "Please fix it and return ONLY the corrected, valid JSON string. "
        "Do not add any explanations, apologies, or markdown formatting like ```json ... ```.\n\n"
        f"Malformed JSON attempt:\n```\n{pre_repaired_str}\n```\n\nCorrected JSON string:"
    )
    
    ai_fixed_str = call_ai(
        config, 
        config.json_fixer_model_name,
        [{"role": "user", "content": fixer_prompt}],
        max_tokens_output=max(2048, int(len(pre_repaired_str) * 1.5)), # 分配足够的空间
        temperature=0.0
    )

    if "AI模型调用失败" in ai_fixed_str or not ai_fixed_str.strip():
        logging.error(f"  AI JSON 修复调用失败或返回空。")
        return None

    # 对 AI 修复后的结果再进行一次预处理和解析
    final_attempt_str = preprocess_json_string(ai_fixed_str)
    if (parsed_str := _try_parse(final_attempt_str, "AI 修复后")) is not None:
        return parsed_str

    logging.error(f"在所有三个阶段（直接、预处理、AI修复）后，都无法从响应中解析出有效的 JSON。")
    return None

def extract_knowledge_gaps(feedback: str) -> list[str]:
    """从审稿人的反馈中提取知识空白列表。"""
    # 使用正则表达式匹配 'KNOWLEDGE GAPS' 部分，不区分大小写，并处理多行内容
    match = re.search(
        r'###?\s*KNOWLEDGE GAPS\s*###?\s*\n(.*?)(?=\n###?|\Z)', 
        feedback, 
        re.DOTALL | re.IGNORECASE
    )
    if not match:
        logging.info("反馈中未找到 'KNOWLEDGE GAPS' 部分。")
        return []
        
    content = match.group(1).strip()
    # 按数字、- 或 * 分割列表项
    gaps = [g.strip() for g in re.split(r'\n\s*(?:\d+\.|\-|\*)\s*', content) if g.strip()]
    
    logging.info(f"从反馈中提取了 {len(gaps)} 个知识空白。")
    return gaps

def chunk_document_for_rag(config: Config, document_text: str, document_outline_data: dict, doc_id: str) -> tuple[list[str], list[dict]]:
    """
    根据文档大纲结构为 RAG 对文档进行分块。
    每个块都关联元数据，将其链接到其章节和子章节。
    """
    logging.info(f"  正在基于大纲为 RAG 对文档 (doc_id: {doc_id}) 进行分块...")
    chunks, metadatas = [], []
    
    if not document_text or not document_outline_data or not document_outline_data.get("outline"):
        logging.warning("  chunk_document_for_rag: 文档文本或大纲为空/无效。返回空块。")
        return [], []

    # 遍历大纲中的主要章节
    outline = document_outline_data.get("outline", [])
    for i, chapter_item in enumerate(outline):
        chapter_title = chapter_item.get("title", f"未命名章节 {i+1}")
        escaped_chapter_title = re.escape(chapter_title)

        # 确定当前章节的文本范围：从当前章节标题开始，到下一个章节标题或文档末尾结束
        next_chapter_title = outline[i+1].get("title") if i + 1 < len(outline) else None
        
        pattern = rf"(##\s*{escaped_chapter_title}.*?)"
        if next_chapter_title:
            pattern += rf"(?=##\s*{re.escape(next_chapter_title)})"
        
        match = re.search(pattern, document_text, re.DOTALL | re.IGNORECASE)
        
        if not match:
            logging.warning(f"  在文档中未能定位到章节 '{chapter_title}' 的内容用于分块。")
            continue
            
        chapter_content = match.group(1).strip()
        
        # 将章节内容分割成更小的块
        # 使用基于字符的简单滑动窗口分割
        max_chars = int(config.max_chunk_tokens * 2.5) # 基于 token 估算字符
        step = max_chars - config.overlap_chars

        if len(chapter_content) <= max_chars:
            text_chunks = [chapter_content]
        else:
            text_chunks = [chapter_content[j:j+max_chars] for j in range(0, len(chapter_content), step)]

        for chunk_idx, text_chunk in enumerate(text_chunks):
            chunks.append(text_chunk)
            metadatas.append({
                "doc_id": doc_id,
                "chapter_title": chapter_title,
                "chunk_index_in_chapter": chunk_idx
            })

    logging.info(f"  文档 RAG 分块完成。共生成 {len(chunks)} 个块。")
    return chunks, metadatas