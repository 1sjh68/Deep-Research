# utils/file_handler.py

import os
import json
import logging
import fitz  # PyMuPDF
from datetime import datetime 
# 从重构后的模块中导入依赖
from config import Config

def load_external_data(config: Config, file_paths: list[str]) -> str:
    """
    从给定的文件路径列表（支持 .txt 和 .pdf）加载所有文本内容。
    """
    if not file_paths:
        return ""
    
    all_content = []
    for fp in file_paths:
        if not fp or not os.path.exists(fp):
            logging.warning(f"外部数据文件未找到，已跳过: {fp}")
            continue
        
        ext = os.path.splitext(fp)[1].lower()
        content = ""
        try:
            if ext == '.txt':
                with open(fp, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif ext == '.pdf':
                with fitz.open(fp) as doc:
                    content = "".join(page.get_text() for page in doc)
            else:
                logging.warning(f"不支持的文件类型: {ext} (文件: {fp})。已跳过。")
                continue
            
            logging.info(f"成功读取 {ext.upper()} 文件: {fp} ({len(content)} 字符)")
            # 为每个文件内容添加明确的分隔符和元信息
            file_header = f"\n--- 文件开始: {os.path.basename(fp)} ---\n"
            file_footer = f"\n--- 文件结束: {os.path.basename(fp)} ---\n"
            all_content.append(file_header + content + file_footer)
            
        except Exception as e:
            logging.error(f"读取外部数据文件 {fp} 时出错: {e}")
            
    return "\n".join(all_content)

def save_checkpoint(config: Config, iteration: int, solution: str, feedback_history: list,
                    initial_problem: str, initial_solution_target_chars: int,
                    max_iterations: int, external_data_checksum: str,
                    document_outline_data: dict | None = None,
                    successful_patches: list[dict] | None = None,
                    research_briefs_history: list[str] | None = None,
                    style_guide: str | None = None
                    ):
    """
    将当前任务状态序列化为 JSON 并保存到检查点文件。
    """
    checkpoint_data = {
        "metadata": {
            "version": "1.1", # 可以加入版本号以便未来迁移
            "timestamp": datetime.now().isoformat()
        },
        "state": {
            "iteration": iteration,
            "current_solution": solution,
            "feedback_history": feedback_history,
            "initial_problem": initial_problem,
            "initial_solution_target_chars": initial_solution_target_chars,
            "max_iterations": max_iterations,
            "external_data_checksum": external_data_checksum,
            "document_outline_data": document_outline_data,
            "successful_patches": successful_patches if successful_patches else [],
            "research_briefs_history": research_briefs_history if research_briefs_history else [],
            "style_guide": style_guide
        }
    }
    
    # 确保会话目录存在
    if not config.session_dir:
        logging.error("会话目录未设置，无法保存检查点。")
        return

    path = os.path.join(config.session_dir, config.checkpoint_file_name)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=4)
        logging.info(f"\n--- 检查点已保存至 {path} (迭代 {iteration + 1} 完成) ---")
    except Exception as e:
        logging.error(f"\n保存检查点时出错: {e}")

def load_checkpoint(config: Config):
    """
    从检查点文件加载任务状态。
    """
    # 确保会话目录存在
    if not config.session_dir:
        logging.error("会话目录未设置，无法加载检查点。")
        return None
        
    path = os.path.join(config.session_dir, config.checkpoint_file_name)
    if not os.path.exists(path):
        return None
        
    try:
        with open(path, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
            
        # 兼容旧格式，同时提取 state 数据
        if "state" in checkpoint_data:
            state = checkpoint_data["state"]
        else:
            state = checkpoint_data # 假设是旧格式

        logging.info(f"\n--- 成功从 {path} 加载检查点 (保存于迭代 {state.get('iteration', -1) + 1} 后) ---")
        
        # 为可能缺失的键提供默认值，以增强向后兼容性
        state.setdefault("successful_patches", [])
        state.setdefault("research_briefs_history", [])
        state.setdefault("style_guide", "")
        
        return state
        
    except (json.JSONDecodeError, KeyError) as e:
        logging.error(f"\n加载检查点时出错: {e}。可能文件已损坏或格式不兼容，将删除。")
        try:
            os.remove(path)
        except Exception as remove_err:
            logging.error(f"删除损坏的检查点文件时出错: {remove_err}")
    except Exception as e:
        logging.error(f"\n加载检查点时发生未知错误: {e}", exc_info=True)
        
    return None

def delete_checkpoint(config: Config):
    """
    删除检查点文件，通常在任务成功结束后调用。
    """
    if not config.session_dir:
        return

    path = os.path.join(config.session_dir, config.checkpoint_file_name)
    if os.path.exists(path):
        try:
            os.remove(path)
            logging.info(f"\n--- 任务完成，检查点文件 {path} 已删除 ---")
        except Exception as e:
            logging.error(f"\n删除检查点文件时出错: {e}")