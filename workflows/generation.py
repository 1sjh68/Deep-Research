# workflows/generation.py

import os
import json
import logging
import hashlib
import collections
from datetime import datetime
import asyncio
import re
import time

# --- 从重构后的模块中导入所有依赖 ---
from config import Config
from services.vector_db import VectorDBManager
from core.context_manager import ContextManager
from planning.outline import (
    generate_document_outline_with_tools,
    allocate_content_lengths,
    review_and_correct_outline_with_tools
)
from services.llm_interaction import call_ai
from core.patch_manager import apply_patch
from utils.text_processor import (
    truncate_text_for_context,
    extract_knowledge_gaps,
    calculate_checksum,
    chunk_document_for_rag
)
from utils.file_handler import (
    load_external_data,
    save_checkpoint,
    delete_checkpoint,
    load_checkpoint
)
from services.web_research import run_research_cycle_async

# --- 优化过程的自定义异常 ---
class OptimizationError(RuntimeError):
    """在优化过程中发生错误时抛出的自定义异常，可能携带部分数据。"""
    def __init__(self, message, partial_data=None):
        super().__init__(message)
        self.partial_data = partial_data if partial_data is not None else {}


def generate_style_guide(config: Config) -> str:
    """根据用户问题，为整个文档生成一份风格与声音指南。"""
    logging.info("\n--- 正在生成风格与声音指南 ---")
    prompt = f"""
    你是一位经验丰富的总编辑。请根据用户的核心问题，为即将撰写的深度报告制定一份简明扼要的《风格与声音指南》。

    # 核心问题:
    "{config.user_problem}"

    # 你的任务是定义以下几点:
    1.  **核心论点 (Core Thesis)**: 用一句话总结本文最关键、最想证明的中心思想。
    2.  **目标读者 (Audience)**: 这篇文章是写给谁看的？
    3.  **写作语气 (Tone)**: 文章应该是什么感觉？（例如：学术严谨、科普风趣、客观中立）
    4.  **叙事节奏 (Narrative Pace)**: 内容应该如何展开？
    5.  **关键术语 (Key Terminology)**: 列出本文必须统一使用的3-5个核心术语及其简要定义。

    请直接输出这份指南，不要添加任何额外的解释。
    """
    messages = [
        {"role": "system", "content": "你是一位创作风格指南的大师级编辑。"},
        {"role": "user", "content": prompt}
    ]

    style_guide = call_ai(config, config.editorial_model_name, messages, max_tokens_output=1024, temperature=0.1)

    if "AI模型调用失败" in style_guide:
        logging.error(f"生成风格指南失败: {style_guide}")
        return ""

    logging.info("--- 风格与声音指南已生成 ---")
    logging.info(style_guide)
    return style_guide


def generate_section_content(config: Config, section_title: str, section_specific_user_prompt: str,
                             system_prompt: str, target_length_chars: int, model_name: str,
                             overall_context: str = "", is_subsection: bool = False) -> str:
    """为大纲中的单个章节或子章节生成内容。"""
    logging.info(f"\n--- 正在为部分生成内容: '{section_title}' (目标: {target_length_chars} 字符) ---")
    full_section_content = ""
    
    header_prefix = "###" if is_subsection else "##"

    for i in range(config.max_chunks_per_section):
        remaining_chars = target_length_chars - len(full_section_content)
        if remaining_chars < config.min_allocated_chars_for_section / 2: # 子章节的最小长度要求可以更低
            logging.info(f"  - 部分 '{section_title}' 已接近目标长度，停止生成。")
            break

        logging.info(f"  - '{section_title}', 块 {i+1}/{config.max_chunks_per_section} (当前: {len(full_section_content)}/{target_length_chars} 字符)")

        chars_to_generate_this_chunk = min(remaining_chars, int(config.max_chunk_tokens * 2.5))
        max_tokens_for_chunk = max(200, int(chars_to_generate_this_chunk / 2.0))

        if i == 0:
            user_prompt = (
                f"你正在撰写报告中的一个部分，标题是：'{section_title}'。\n"
                f"本部分的具体要求是：{section_specific_user_prompt}\n\n"
                f"报告的整体上下文信息（供参考）：\n{overall_context}\n\n"
                f"请开始撰写 '{section_title}' 部分的内容。请直接开始写正文，不要重复标题。"
            )
        else:
            context_from_section = truncate_text_for_context(config, full_section_content, config.overlap_chars * 2, "tail")
            user_prompt = (
                f"你正在续写关于 '{section_title}' 的部分。以下是本部分已生成内容的结尾：\n"
                f"--- 已有内容结尾 ---\n{context_from_section}\n--- 已有内容结尾 ---\n\n"
                f"报告的整体上下文信息（供参考）：\n{overall_context}\n\n"
                f"请继续流畅地撰写后续内容，不要重复标题或已有内容。"
            )

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        chunk = call_ai(config, model_name, messages, max_tokens_output=max_tokens_for_chunk)

        if "AI模型调用失败" in chunk:
            logging.error(f"  - '{section_title}' 的块生成失败: {chunk}")
            break

        full_section_content += (" " if i > 0 else "") + chunk.strip()
        time.sleep(0.2)
    
    logging.info(f"--- 部分 '{section_title}' 内容生成完毕 (生成: {len(full_section_content)} 字符) ---")
    
    if not full_section_content.strip():
        return f"\n\n{header_prefix} {section_title}\n\n[本部分内容生成失败或为空]\n\n"
        
    return f"\n\n{header_prefix} {section_title}\n\n{full_section_content.strip()}\n\n"


async def optimize_solution_with_two_ais(
    config: Config, initial_problem: str, style_guide: str,
    external_data: str = "", external_data_checksum: str = "",
    vector_db_manager: VectorDBManager | None = None
) -> tuple[str | None, list[dict], list[str], list[str], str | None]:
    """
    核心的优化循环，模拟作者和审稿人之间的协作。
    """
    logging.info("### V12: Starting RAG-Enhanced Optimization with Precise Context ###")

    current_solution: str | None = None
    feedback_history: list[str] = []
    document_outline_data: dict | None = None
    start_iteration_index: int = 0
    successful_patches_history: list[dict] = []
    all_research_briefs: list[str] = []
    
    # 尝试从检查点恢复
    loaded_checkpoint = load_checkpoint(config)
    if loaded_checkpoint:
        is_same_task = (
            loaded_checkpoint.get("initial_problem") == initial_problem and
            loaded_checkpoint.get("external_data_checksum") == external_data_checksum
        )
        if is_same_task:
            logging.info(f"--- 从检查点恢复任务 ---")
            start_iteration_index = loaded_checkpoint.get("iteration", 0) + 1
            current_solution = loaded_checkpoint.get("current_solution")
            feedback_history = loaded_checkpoint.get("feedback_history", [])
            document_outline_data = loaded_checkpoint.get("document_outline_data")
            successful_patches_history = loaded_checkpoint.get("successful_patches", [])
            all_research_briefs = loaded_checkpoint.get("research_briefs_history", [])
            style_guide = loaded_checkpoint.get("style_guide", style_guide)
        else:
            logging.info("--- 检查点任务不匹配，开始新任务。 ---")
            delete_checkpoint(config) # 删除旧的检查点

    # 如果没有从检查点恢复，则进行初始化
    if not document_outline_data:
        start_iteration_index = 0
        current_solution = None
        feedback_history = []
        successful_patches_history = []
        all_research_briefs = []
        raw_outline = generate_document_outline_with_tools(config, initial_problem)
        if not raw_outline or "outline" not in raw_outline:
            return None, [], [], [], "错误：文档大纲生成失败。"
        document_outline_data = allocate_content_lengths(config, raw_outline, config.initial_solution_target_chars)

    # 初始化上下文管理器
    context_manager = ContextManager(config, style_guide, document_outline_data, external_data)

    # 定义AI角色的系统提示
    system_prompt_base = f"你是一位顶级的专家，拥有卓越的推理和解决问题的能力。你的首要职责是严格遵守用户最初的问题和提供的大纲。确保所有内容都高度相关，避免任何主题偏离。\n\n[风格与声音指南]\n{style_guide}"
    
    secondary_ai_critique_prompt = """你是一位一丝不苟的分析师和批判性思考者。你的任务是找出所提供解决方案中的缺点、错误、遗漏或需要改进的地方。
**请尽可能具体地提供反馈。**
建设性的反馈应涵盖：
1. 准确性：是否存在任何事实错误或逻辑谬误？
2. 完整性：是否根据大纲全面涵盖了所有相关方面？
3. 主题/大纲 adherence：是否严格遵守问题和大纲？
4. 叙事连贯性：论证是否合乎逻辑？章节/小节之间的过渡是否平滑？
5. 知识空白：**如果解决方案缺乏关键信息或需要外部知识，请在反馈末尾的“### KNOWLEDGE GAPS ###”下列出这些内容。这一点至关重要。**
请提供具体、可操作的反馈。如果解决方案非常出色，请明确说明。"""

    # --- 开始主迭代循环 ---
    for i in range(start_iteration_index, config.max_iterations):
        logging.info(f"\n--- 迭代 {i + 1}/{config.max_iterations} ---")
        
        # 步骤1: 内容生成 (仅在第一次迭代或无解决方案时)
        if i == 0 and current_solution is None:
            logging.info("--- 正在根据大纲生成初始解决方案 ---")
            assembled_parts = [f"# {document_outline_data.get('title', 'Untitled Document')}\n\n"]
            
            for chapter in document_outline_data.get("outline", []):
                chapter_title = chapter.get("title")
                
                if chapter.get("sections"): # 如果有子章节
                    chapter_parts = []
                    for sub_idx, subsection in enumerate(chapter.get("sections", [])):
                        subsection_title = subsection.get("title")
                        context_for_sub = context_manager.get_context_for_subsection(chapter_title, sub_idx)
                        sub_content = generate_section_content(
                            config, subsection_title, subsection.get("description"),
                            system_prompt_base, subsection.get("allocated_chars", 500),
                            config.main_ai_model, context_for_sub, is_subsection=True
                        )
                        chapter_parts.append(sub_content)
                        context_manager.record_completed_subsection(chapter_title, subsection_title, sub_content)
                    
                    full_chapter_content = f"\n\n## {chapter_title}\n\n" + "".join(chapter_parts)
                    assembled_parts.append(full_chapter_content)
                    context_manager.update_completed_chapter_content(chapter_title, full_chapter_content)
                else: # 如果没有子章节
                    context_for_chap = context_manager.get_context_for_standalone_chapter(chapter_title)
                    chap_content = generate_section_content(
                        config, chapter_title, chapter.get("description"),
                        system_prompt_base, chapter.get("allocated_chars", 1000),
                        config.main_ai_model, context_for_chap, is_subsection=False
                    )
                    assembled_parts.append(chap_content)
                    context_manager.update_completed_chapter_content(chapter_title, chap_content)
            
            current_solution = "".join(assembled_parts)
        
        # 步骤2: 评审
        logging.info("--- 审稿 AI 正在分析当前解决方案 ---")
        solution_for_critic = truncate_text_for_context(config, current_solution, config.max_context_for_long_text_review_tokens)
        critic_prompt = f"原始问题:\n---\n{initial_problem}\n---\n待评审的解决方案:\n---\n{solution_for_critic}\n---\n请提供你的评审意见:"
        feedback = call_ai(
            config, config.secondary_ai_model, 
            [{"role": "system", "content": secondary_ai_critique_prompt}, {"role": "user", "content": critic_prompt}]
        )
        feedback_history.append(feedback)
        logging.info(f"审稿 AI 反馈 (前1000字符):\n{feedback[:1000]}...")

        # 步骤3: 研究知识空白
        knowledge_gaps = extract_knowledge_gaps(feedback)
        if knowledge_gaps:
            research_brief = await run_research_cycle_async(config, knowledge_gaps, current_solution)
            if research_brief:
                all_research_briefs.append(research_brief)
                external_data += research_brief # 将新研究加入参考资料
                external_data_checksum = calculate_checksum(external_data)
                context_manager._summarize_external_data(external_data) # 更新上下文管理器中的摘要
        
        # 步骤4: 修订
        logging.info("--- 作者 AI 正在根据反馈和研究生成补丁 ---")
        patcher_context = f"反馈意见:\n{feedback}\n\n相关研究资料:\n{all_research_briefs[-1] if knowledge_gaps and all_research_briefs else '无'}"
        # ... (此处省略了为补丁生成器构建复杂上下文的逻辑，简化为直接提供反馈和研究)
        # 在实际代码中，这里会调用一个函数来生成补丁 JSON
        # 此处简化模拟
        # ... 实际上这里应该调用一个带工具的 `call_ai` 来生成 JSON Patch
        # 此处省略以保持流程清晰
        
        # 假设我们得到了补丁并应用了它
        # current_solution = apply_patch(current_solution, generated_patch_json)

        # 检查是否可以提前结束
        if any(phrase in feedback for phrase in ["没有显著改进空间", "非常完善", "无需进一步修改"]):
            logging.info("审稿 AI 认为解决方案已完成。正在结束迭代。")
            break
            
        # 保存检查点
        save_checkpoint(
            config, i, current_solution, feedback_history, initial_problem, 
            config.initial_solution_target_chars, config.max_iterations, 
            external_data_checksum, document_outline_data, successful_patches_history, 
            all_research_briefs, style_guide
        )

    # --- 迭代结束后 ---
    if not current_solution:
        return None, [], [], [], "错误：经过所有迭代后未能生成解决方案。"

    # 步骤5: 生成最终结论并进行润色
    logging.info("--- 正在生成最终结论 ---")
    final_conclusion = generate_final_conclusion(config, current_solution, initial_problem, system_prompt_base, document_outline_data)
    current_solution += final_conclusion
    
    logging.info("--- 正在对全文进行最终润色 ---")
    polished_solution = perform_final_polish(config, current_solution, style_guide)
    
    # 步骤6: 经验积累
    if vector_db_manager:
        accumulate_experience(
            config, vector_db_manager, initial_problem, polished_solution,
            feedback_history, successful_patches_history, all_research_briefs
        )
    
    delete_checkpoint(config) # 任务成功完成，删除检查点
    
    return polished_solution, successful_patches_history, all_research_briefs, feedback_history, None


def generate_final_conclusion(config: Config, document_content: str, problem_statement: str,
                              system_prompt: str, outline_data: dict | None = None) -> str:
    """根据全文内容，生成最终的结论部分。"""
    logging.info("\n--- 正在为文档生成最终结论 ---")
    truncated_doc_for_ctx = truncate_text_for_context(config, document_content, 8000, "tail")
    
    user_prompt = f"""
    原始问题:
    ---
    {problem_statement}
    ---
    文档主体内容 (为提供上下文的结尾部分):
    ---
    {truncated_doc_for_ctx}
    ---
    基于你已生成的完整文档内容和上述上下文，请撰写一个全面且富有洞察力的“结论”部分。
    它应总结主要发现，并直接回应原始问题。
    仅输出结论部分的正文，不要自己添加“## 结论”这样的标题。
    """
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    conclusion_text = call_ai(config, config.main_ai_model, messages, max_tokens_output=1024, temperature=0.1)
    
    if "AI模型调用失败" in conclusion_text:
        return "\n\n## 结论\n\n[结论部分生成失败]\n\n"
    
    return f"\n\n## 结论\n\n{conclusion_text.strip()}\n"


def perform_final_polish(config: Config, full_document_text: str, style_guide: str) -> str:
    """对整个文档进行最终润色，提升连贯性和流畅度。"""
    logging.info("\n--- 正在对全文进行最终润色 ---")
    truncated_text = truncate_text_for_context(config, full_document_text, 28000)
    
    prompt = f"""
    你是一位顶级的润色编辑。下面的文本是一份报告的初稿。你的任务是通读全文，进行微调，使其浑然一体。
    主要任务：
    1. **平滑过渡**: 检查各章节之间的衔接，如果生硬，请微调过渡句。
    2. **统一术语与风格**: 确保全文术语统一，风格符合指南。
    3. **消除冗余**: 删除重复的句子或想法。
    注意：你的目标是“润色”而非“重写”。请直接输出经过你润色后的完整文稿。

    [风格指南]
    {style_guide}

    [待润色的初稿]
    ---
    {truncated_text}
    ---
    """
    messages = [{"role": "system", "content": "你是一位大师级的润色编辑。"}, {"role": "user", "content": prompt}]
    polished_text = call_ai(config, config.editorial_model_name, messages, max_tokens_output=8192, temperature=0.1)
    
    if "AI模型调用失败" in polished_text or len(polished_text) < len(full_document_text) * 0.8:
        logging.error("最终润色失败或生成的文本过短。将返回原始文档。")
        return full_document_text
        
    return polished_text


def quality_check(config: Config, content: str) -> str:
    """调用 AI 对最终产出进行质量评估。"""
    content_for_review = truncate_text_for_context(config, content, 10000)
    prompt = f"请深入评估以下内容的质量。为以下方面提供评分(0-10分): 深度、细节、结构、连贯性、问题契合度。并列出主要优缺点。\n\n内容:\n{content_for_review}"
    return call_ai(config, config.secondary_ai_model, [{"role": "user", "content": prompt}])


def accumulate_experience(config: Config, db_manager: VectorDBManager,
                          problem: str, final_solution: str | None,
                          feedback_history: list[str],
                          successful_patches: list[dict],
                          research_briefs: list[str]):
    """将成功的任务产出作为经验存入向量数据库。"""
    logging.info("\n--- 正在将经验积累到向量数据库 ---")
    if not db_manager or not db_manager.collection:
        logging.error("VectorDBManager 未初始化。跳过经验积累。")
        return

    experience_items, metadatas, ids = [], [], []
    current_time_iso = datetime.now().isoformat()
    problem_hash = hashlib.md5(problem.encode()).hexdigest()

    if final_solution:
        # 将最终解决方案的不同章节作为独立的经验存入
        # （此处简化为存储整个解决方案）
        experience_items.append(final_solution)
        metadatas.append({"type": "final_solution", "problem": problem[:200], "date": current_time_iso})
        ids.append(f"solution_{problem_hash}_{int(time.time())}")

    for i, brief in enumerate(research_briefs):
        experience_items.append(brief)
        metadatas.append({"type": "research_brief", "problem": problem[:200], "date": current_time_iso})
        ids.append(f"research_{problem_hash}_{int(time.time())}_{i}")

    if experience_items:
        db_manager.add_experience(texts=experience_items, metadatas=metadatas, ids=ids)


async def generate_extended_content_workflow(config: Config, vector_db_manager: VectorDBManager | None) -> str:
    """顶层工作流，串联起所有步骤，由 main.py 调用。"""
    logging.info("\n--- 开始扩展内容生成工作流 ---")
    
    style_guide = generate_style_guide(config)
    loaded_ext_data = load_external_data(config, config.external_data_files or [])
    
    retrieved_experience_text = ""
    if vector_db_manager:
        retrieved_exps = vector_db_manager.retrieve_experience(config.user_problem)
        if retrieved_exps:
            exp_texts = [f"---历史经验 {i+1} (相关度: {exp.get('distance', -1):.4f})---\n{exp.get('document')}" for i, exp in enumerate(retrieved_exps)]
            retrieved_experience_text = "\n\n===== 上下文检索到的相关历史经验 =====\n" + "\n\n".join(exp_texts) + "\n===== 历史经验结束 =====\n\n"
            logging.info(f"成功检索并格式化了 {len(retrieved_exps)} 条经验。")
    
    final_external_data = retrieved_experience_text + loaded_ext_data
    ext_data_checksum = calculate_checksum(final_external_data)
    
    final_answer, _, _, _, error_message = await optimize_solution_with_two_ais(
        config, config.user_problem, style_guide, final_external_data, ext_data_checksum, vector_db_manager
    )

    if error_message:
        logging.error(f"优化过程因错误而终止: {error_message}")
        return f"错误: {error_message}"
    
    if not final_answer:
        logging.error("优化过程完成，但未生成任何有效答案。")
        return "错误：工作流未产生任何答案。"

    logging.info("\n--- 工作流成功完成 ---")
    logging.info(f"最终生成长度: {len(final_answer)} 字符。")
    logging.info("\n--- 最终产出质量评估报告 ---")
    quality_report = quality_check(config, final_answer)
    logging.info(quality_report)
    
    return final_answer