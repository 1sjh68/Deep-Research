# services/llm_interaction.py

import openai
import logging
import time
import re
import json
import tenacity

# 从重构后的模块中导入依赖
from config import Config

class EmptyResponseFromReasonerError(Exception):
    """当 Reasoner 模型在剥离 <think> 标签后返回空内容时抛出的自定义异常。"""
    pass

def call_ai_core(config: Config, model_name: str, messages: list, temperature: float,
                 effective_max_output_tokens: int, top_p: float,
                 frequency_penalty: float, presence_penalty: float) -> str:
    """
    核心 AI 调用逻辑，由 tenacity 包装以实现重试。
    """
    client = config.client
    start_time = time.perf_counter()
    
    response = client.chat.completions.create(
        model=model_name, messages=messages, temperature=temperature,
        max_tokens=effective_max_output_tokens, top_p=top_p,
        frequency_penalty=frequency_penalty, presence_penalty=presence_penalty
    )
    duration = time.perf_counter() - start_time
    content = response.choices[0].message.content
    
    final_content = content
    # 检查是否为需要特殊处理<think>标签的模型
    is_reasoner_model_for_stripping = (
        model_name == config.main_ai_model_heavy or
        model_name == config.secondary_ai_model
    )

    if is_reasoner_model_for_stripping and content:
        think_match = re.search(r"<think>([\s\S]*?)</think>", content, re.DOTALL)
        if think_match:
            thought_process = think_match.group(1).strip()
            logging.info(f"    - [DeepSeek Reasoner] 提取到思考过程 ({len(thought_process)} chars): {thought_process[:500]}...")
            final_content = content.replace(think_match.group(0), "").strip()
            logging.info(f"    - [DeepSeek Reasoner] 剥离 <think> 后的内容 ({len(final_content)} chars). 预览: {final_content[:300]}...")
        else:
            logging.info(f"    - [DeepSeek Reasoner] 响应中未找到 <think> 块。")
    
    total_tokens = response.usage.total_tokens if response.usage else 'N/A'
    logging.info(f"    - API 调用成功 ({duration:.2f}s), 模型: {model_name}, 最终内容长度: {len(final_content) if final_content else 0} chars, 总 Tokens: {total_tokens}.")

    if not final_content or final_content.isspace():
        logging.warning(f"    - AI 调用返回空或纯空白内容 (模型: {model_name})")
        if is_reasoner_model_for_stripping:
            # 抛出自定义异常，以便 tenacity 可以捕获并重试
            raise EmptyResponseFromReasonerError(f"模型 {model_name} 在剥离 <think> 标签后返回空内容。")
            
    return final_content if final_content else ""


def call_ai(config: Config, model_name: str, messages: list, temperature: float = 0.05,
            max_tokens_output: int = -1, top_p: float = 0.95, frequency_penalty: float = 0.2,
            presence_penalty: float = 0.05) -> str:
    """
    带健壮重试机制和智能 token 管理的 AI 调用封装函数。
    """
    # 模型上下文限制 (tokens)
    model_context_limits = {
        "deepseek-reasoner": 64000,
        "deepseek-chat": 64000,
        "deepseek-coder": 128000,
    }
    # 为配置中的所有模型角色设置默认值
    for model_attr in [
        'main_ai_model', 'main_ai_model_heavy', 'secondary_ai_model', 'summary_model_name',
        'researcher_model_name', 'outline_model_name', 'planning_review_model_name',
        'editorial_model_name', 'json_fixer_model_name'
    ]:
        m_name = getattr(config, model_attr, None)
        if m_name and m_name not in model_context_limits:
            model_context_limits[m_name] = model_context_limits.get(m_name, 64000) # 默认64k

    # 不同模型的默认最大输出 token
    model_specific_max_output = {
        "deepseek-reasoner": 4096,
        "deepseek-chat": 8192,
        "deepseek-coder": 8192,
    }
    # 为配置中的所有模型角色设置默认值
    for model_attr in [
        'main_ai_model', 'main_ai_model_heavy', 'secondary_ai_model', 'summary_model_name',
        'researcher_model_name', 'outline_model_name', 'planning_review_model_name',
        'editorial_model_name', 'json_fixer_model_name'
    ]:
        m_name = getattr(config, model_attr, None)
        if m_name and m_name not in model_specific_max_output:
            # 推理模型给稍低的默认值，编码和聊天模型给高一些
            default_max = 4096 if 'reasoner' in m_name else 8192
            model_specific_max_output[m_name] = model_specific_max_output.get(m_name, default_max)
    
    # 决定有效的最大输出 token 数
    if max_tokens_output > 0:
        effective_max_output_tokens = min(max_tokens_output, model_specific_max_output.get(model_name, 4096))
    else:
        effective_max_output_tokens = model_specific_max_output.get(model_name, 4096)

    # 确保推理模型有足够的空间进行 CoT (Chain-of-Thought)
    is_reasoner_model = 'reasoner' in model_name
    reasoner_min_tokens = 2048
    if is_reasoner_model and effective_max_output_tokens < reasoner_min_tokens:
        logging.info(f"    - 为 {model_name} 调整 max_tokens_output 至 {reasoner_min_tokens} (以容纳 CoT)。")
        effective_max_output_tokens = reasoner_min_tokens
        
    # 计算输入 token 并检查是否可能超出上下文
    total_input_tokens = sum(config.count_tokens(m['content']) for m in messages)
    logging.info(f"    - AI 调用: 模型={model_name}, 输入 Tokens (估算): {total_input_tokens}, 请求输出 Tokens: {max_tokens_output} -> 有效最大值: {effective_max_output_tokens}")

    model_context_limit = model_context_limits.get(model_name, 64000)
    if total_input_tokens + effective_max_output_tokens > model_context_limit:
        logging.warning(f"    - 警告: 输入+输出 Tokens ({total_input_tokens + effective_max_output_tokens}) 可能超过模型 {model_name} 的上下文限制 ({model_context_limit})。")
        available_for_output = model_context_limit - total_input_tokens
        if available_for_output < effective_max_output_tokens:
            new_max_output = max(100, available_for_output - 100) # 至少保留100 tokens 并留一些余地
            logging.info(f"    - 调整 max_tokens_output 从 {effective_max_output_tokens} 到 {new_max_output} 以适应上下文。")
            effective_max_output_tokens = new_max_output
        if effective_max_output_tokens <= 0:
            logging.error(f"    - 严重错误: 模型 {model_name} 没有可用的输出 Tokens。输入 Tokens: {total_input_tokens}, 上下文限制: {model_context_limit}")
            return "AI模型调用失败 (错误): 输入内容已占满上下文窗口，无法生成回复。"

    # 配置重试逻辑
    retryer = tenacity.Retrying(
        wait=tenacity.wait_exponential(
            multiplier=config.api_retry_wait_multiplier,
            min=2,
            max=config.api_retry_max_wait
        ),
        stop=tenacity.stop_after_attempt(config.api_retry_max_attempts),
        retry=(
            tenacity.retry_if_exception_type(openai.APITimeoutError) |
            tenacity.retry_if_exception_type(openai.APIConnectionError) |
            tenacity.retry_if_exception_type(openai.InternalServerError) |
            tenacity.retry_if_exception_type(openai.RateLimitError) |
            tenacity.retry_if_exception_type(EmptyResponseFromReasonerError)
        ),
        before_sleep=tenacity.before_sleep_log(logging.getLogger(__name__), logging.WARNING),
        reraise=True
    )

    try:
        return retryer(
            call_ai_core, config, model_name, messages, temperature,
            effective_max_output_tokens, top_p, frequency_penalty, presence_penalty
        )
    except openai.APIStatusError as e:
        logging.error(f"    - 模型 {model_name} 的 API 调用状态错误 (未重试或最终尝试失败): {e.status_code} - {e.response.text if e.response else '无响应文本'}")
        if e.status_code == 400:
            logging.error(f"提示：请求可能无效 (例如，输入Token {total_input_tokens} + 输出 {effective_max_output_tokens} 超出模型限制)。这是一个不可重试的客户端错误。")
        error_message_detail = "未知错误"
        if e.response is not None:
            try:
                error_message_detail = e.response.json().get('error',{}).get('message','未知错误')
            except json.JSONDecodeError:
                error_message_detail = e.response.text if e.response.text else "无响应文本"
        return f"AI模型调用失败 (API 错误 {e.status_code}): {error_message_detail}"
    except Exception as e:
        logging.error(f"    - 模型 {model_name} 的 AI 调用因未处理的异常或所有重试后失败: {e}", exc_info=True)
        return "AI模型调用失败，请检查网络连接或API密钥/设置，或查看详细日志。"