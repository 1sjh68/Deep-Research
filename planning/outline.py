# planning/outline.py
import os 
import logging
import json

# 从重构后的模块中导入依赖
from config import Config
from services.llm_interaction import call_ai
from planning.tool_definitions import get_initial_outline_tool_definition, get_outline_review_tool_definition

def generate_document_outline_with_tools(config: Config, problem_statement: str) -> dict | None:
    """
    使用强制工具调用功能，生成结构化、可靠的初始文档大綱。
    """
    logging.info(f"\n--- 正在为问题生成文档大纲 (使用工具调用): {problem_statement[:100]}... ---")
    
    tools = get_initial_outline_tool_definition()
    
    outline_prompt = f"""
    分析以下用户请求并创建一个详细的文档大纲。
    最终报告的长度应约为 {config.initial_solution_target_chars} 个字符。
    您必须使用 `create_initial_document_outline` 工具来构建您的响应。

    用户请求: "{problem_statement}"
    """
    messages = [
        {"role": "system", "content": "你是一位结构分析师，使用 `create_initial_document_outline` 工具创建文档大纲。"},
        {"role": "user", "content": outline_prompt}
    ]
    
    try:
        response = config.client.chat.completions.create(
            model=config.outline_model_name,
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "create_initial_document_outline"}},
            temperature=0.05
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if not tool_calls:
            logging.error("generate_document_outline_with_tools: AI 未按预期返回工具调用。")
            return None
        
        function_args_str = tool_calls[0].function.arguments
        parsed_outline_data = json.loads(function_args_str)

        if not parsed_outline_data.get("outline") or not parsed_outline_data.get("title"):
            logging.error(f"从工具调用生成的 JSON 大纲缺少 'title' 或 'outline' 键。获取到: {parsed_outline_data}")
            return None

        num_chapters = len(parsed_outline_data.get('outline', []))
        logging.info(f"--- 通过工具调用成功生成文档大纲 ({num_chapters} 个主要章节) ---")
        
        # 将生成的原始大纲保存到会话目录以供调试
        if config.session_dir:
            path = os.path.join(config.session_dir, "generated_document_outline_tool_call.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(parsed_outline_data, f, ensure_ascii=False, indent=4)
        
        return parsed_outline_data
        
    except Exception as e:
        logging.error(f"  使用工具生成文档大纲时出错: {e}", exc_info=True)
        return None

def review_and_correct_outline_with_tools(config: Config, original_problem: str, completed_summary: str, 
                                          remaining_outline: list, style_guide: str, 
                                          latest_feedback: str = "") -> list | None:
    """
    使用工具调用模式，让 AI 审查并修正剩余的大纲，确保返回可靠的 JSON。
    """
    logging.info("--- V11: 调用战略规划师审查剩余大纲 (使用工具调用) ---")

    tools = get_outline_review_tool_definition()

    style_guide_prompt_block = f"[风格指南]\n{style_guide}\n" if style_guide else ""
    remaining_outline_json = json.dumps(remaining_outline, ensure_ascii=False, indent=2)
    feedback_section = f"\n[最近的专家反馈]\n{latest_feedback}" if latest_feedback else ""
    
    prompt = f"""
你是一位顶级的项目战略规划师，正在对一个复杂的报告项目进行中期复盘。
{style_guide_prompt_block}
[原始目标]
用户的核心需求是："{original_problem}"

[已完成的工作摘要]
---
{completed_summary}
---

[剩余部分的原始计划]
---
{remaining_outline_json}
---
{feedback_section}

[你的任务]
基于已完成的工作和反馈，审慎评估“剩余计划”是否是最佳路径。然后，必须调用 `update_document_outline` 函数来提交最终的、经过优化的剩余计划。
- 如果计划需要调整（例如，章节需要拆分、合并或调整重点），请在调用函数时传入修改后的计划。
- 如果原始计划依然完美，无需改动，请在调用函数时传入与原始计划完全相同的JSON内容。
"""
    messages = [
        {"role": "system", "content": "你是一位战略规划师，必须使用 `update_document_outline` 工具进行响应。"},
        {"role": "user", "content": prompt}
    ]

    try:
        response = config.client.chat.completions.create(
            model=config.planning_review_model_name,
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "update_document_outline"}},
            temperature=0.1
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if not tool_calls:
            logging.error("V11 规划师 (工具) 未按预期返回工具调用。")
            return None
        
        function_args_str = tool_calls[0].function.arguments
        parsed_args = json.loads(function_args_str)
        new_plan = parsed_args.get("revised_plan")

        if not isinstance(new_plan, list):
            logging.error(f"V11 规划师 (工具) 为 'revised_plan' 返回了无效的数据类型 (预期列表，得到 {type(new_plan)})。")
            return None

        # 检查计划是否有实质性变动
        original_plan_str = json.dumps(remaining_outline, sort_keys=True)
        new_plan_str = json.dumps(new_plan, sort_keys=True)
        if original_plan_str == new_plan_str:
            logging.info("V11 规划师 (工具) 审查了计划并确认无需更改。")
            return None # 返回 None 表示无需修改

        logging.info("V11 规划师 (工具) 提出了更新后的大纲。")
        return new_plan

    except Exception as e:
        logging.error(f"V11 规划师 (工具) 失败，出现异常: {e}", exc_info=True)
        return None

def allocate_content_lengths(config: Config, outline_data: dict, total_target_chars: int) -> dict:
    """
    将总目标字数按比例分配给大纲中的每个章节和子章节。
    """
    logging.info(f"\n--- 正在为大纲分配内容长度 (总目标: {total_target_chars} 字符) ---")
    if not outline_data or "outline" not in outline_data or not outline_data["outline"]:
        logging.error("  用于长度分配的大纲数据无效或为空。")
        return outline_data

    outline_items = [item for item in outline_data["outline"] if isinstance(item, dict)]
    if not outline_items:
        logging.error("  大纲中未找到有效的章节项用于长度分配。")
        return outline_data

    # 递归地为所有层级分配字数
    _allocate_recursive(config, outline_items, total_target_chars)

    # 保存分配后的大纲以供调试
    if config.session_dir:
        path = os.path.join(config.session_dir, "allocated_document_outline.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(outline_data, f, ensure_ascii=False, indent=4)
            
    logging.info("--- 内容长度分配完成 ---")
    return outline_data

def _allocate_recursive(config: Config, sections_list: list, parent_allocated_chars: int):
    """
    一个递归的辅助函数，用于将父章节的字数分配给其子章节。
    """
    # 过滤出有效的字典项
    sections_list_dicts = [s for s in sections_list if isinstance(s, dict)]
    if not sections_list_dicts:
        return

    # 1. 标准化比例：处理缺失或无效的比例
    items_with_ratio = []
    items_without_ratio = []
    current_total_ratio = 0.0

    for item in sections_list_dicts:
        ratio = item.get("target_chars_ratio")
        try:
            ratio_float = float(ratio)
            if ratio_float > 0:
                items_with_ratio.append(item)
                item["_ratio_numeric"] = ratio_float
                current_total_ratio += ratio_float
            else:
                items_without_ratio.append(item)
        except (ValueError, TypeError):
            items_without_ratio.append(item)

    # 为没有比例的项分配剩余比例
    if items_without_ratio:
        remaining_ratio = max(0, 1.0 - current_total_ratio)
        ratio_per_item = remaining_ratio / len(items_without_ratio) if items_without_ratio else 0
        for item in items_without_ratio:
            item["_ratio_numeric"] = ratio_per_item

    # 2. 归一化：确保当前层级所有比例总和为 1.0
    final_total_ratio = sum(item.get("_ratio_numeric", 0.0) for item in sections_list_dicts)
    if final_total_ratio > 0:
        for item in sections_list_dicts:
            item["_ratio_numeric"] /= final_total_ratio
    else: # 如果所有比例都为0，则平分
        equal_share = 1.0 / len(sections_list_dicts) if sections_list_dicts else 0
        for item in sections_list_dicts:
            item["_ratio_numeric"] = equal_share

    # 3. 分配字数并处理余数
    total_allocated = 0
    for item in sections_list_dicts:
        item['allocated_chars'] = int(round(item["_ratio_numeric"] * parent_allocated_chars))
        total_allocated += item['allocated_chars']

    remainder = parent_allocated_chars - total_allocated
    # 按比例大小，将余数逐一分配给最大的项，以减少相对误差
    sorted_items = sorted(sections_list_dicts, key=lambda x: x["_ratio_numeric"], reverse=True)
    for i in range(abs(remainder)):
        item_to_adjust = sorted_items[i % len(sorted_items)]
        item_to_adjust['allocated_chars'] += 1 if remainder > 0 else -1

    # 4. 递归调用子章节并清理临时键
    for item in sections_list_dicts:
        logging.info(f"    - {'  ' * item.get('title', '').count('#')}章节 '{item.get('title', 'N/A')}': 分配 {item['allocated_chars']} 字符")
        if "sections" in item and item["sections"]:
            _allocate_recursive(config, item["sections"], item['allocated_chars'])
        del item["_ratio_numeric"] # 清理临时键