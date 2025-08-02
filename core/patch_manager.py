# core/patch_manager.py

import json
import logging
from thefuzz import fuzz

def apply_patch(original_text: str, patches_json_str: str) -> str:
    """
    将一个 JSON 格式的补丁列表应用到原始文本上。
    
    Args:
        original_text: 待修改的完整 Markdown 文本。
        patches_json_str: 一个包含补丁操作列表的 JSON 字符串。

    Returns:
        应用补丁后修改过的文本。
    """
    try:
        # 加载补丁数据，兼容两种可能的格式：一个列表，或者一个包含 "patches" 键的字典
        patches_data = json.loads(patches_json_str)
        if isinstance(patches_data, dict) and "patches" in patches_data:
            patches = patches_data["patches"]
        elif isinstance(patches_data, list):
            patches = patches_data
        else:
            logging.error(f"补丁 JSON 格式无法识别，应为列表或包含 'patches' 键的字典。内容: {patches_json_str[:200]}")
            return original_text
            
    except json.JSONDecodeError as e:
        logging.error(f"补丁字符串中的 JSON 无效: {e}\n补丁内容: {patches_json_str}")
        return original_text
    except Exception as e:
        logging.error(f"加载补丁 JSON 时发生意外错误: {e}\n补丁内容: {patches_json_str}")
        return original_text

    if not patches:
        logging.info("补丁列表为空，无需应用任何更改。")
        return original_text

    # 将文本按行分割，保留换行符，便于操作
    lines = original_text.splitlines(True)
    
    # 遍历每一个补丁操作
    for i, op in enumerate(patches):
        action = op.get("action")
        target_header = op.get("target_section")
        new_content = op.get("new_content", "")
        
        if not action or not target_header:
            logging.warning(f"跳过补丁 {i+1} (缺少 action/target_section): {op}")
            continue
            
        logging.info(f"正在应用补丁 {i+1}: {action.upper()} 到 '{target_header}'")

        # --- 使用模糊匹配查找目标章节标题 ---
        target_idx = -1
        best_match_score = 0
        
        # 仅考虑以 "## " 或 "### " 开头的行作为潜在的标题行，以提高效率和准确性
        potential_header_indices = [
            idx for idx, line in enumerate(lines)
            if line.strip().startswith("## ") or line.strip().startswith("### ")
        ]

        for idx in potential_header_indices:
            line_content = lines[idx].strip()
            score = fuzz.ratio(target_header.strip(), line_content)
            if score > best_match_score:
                best_match_score = score
                target_idx = idx
        
        # 设定一个相似度阈值，防止错误匹配
        SIMILARITY_THRESHOLD = 85
        if best_match_score < SIMILARITY_THRESHOLD:
            logging.warning(f"目标章节 '{target_header}' 未找到或相似度过低 (最佳匹配得分: {best_match_score}%)。跳过此补丁。")
            continue
        
        logging.info(f"  已将目标 '{target_header}' 匹配到第 {target_idx + 1} 行的 '{lines[target_idx].strip()}' (相似度: {best_match_score}%)。")

        # --- 确定章节内容的起止范围 ---
        # 章节结束于下一个同级或更高级别的标题，或者文档末尾
        target_level = lines[target_idx].strip().count('#')
        section_end_idx = len(lines)
        for idx in range(target_idx + 1, len(lines)):
            line = lines[idx].strip()
            if line.startswith("#"):
                current_level = line.count('#')
                if current_level <= target_level:
                    section_end_idx = idx
                    break
        
        # --- 执行补丁操作 ---
        try:
            if action.upper() == "REPLACE":
                # 删除旧内容（标题行之后到章节结束）
                del lines[target_idx + 1 : section_end_idx]
                # 准备并插入新内容
                new_lines = ("\n" + new_content.strip() + "\n").splitlines(True)
                for j, line_to_insert in enumerate(new_lines):
                    lines.insert(target_idx + 1 + j, line_to_insert)
            
            elif action.upper() == "INSERT_AFTER":
                # 在章节末尾插入新内容
                new_lines = ("\n" + new_content.strip() + "\n").splitlines(True)
                # 从后往前插入，以保持正确的顺序
                for line_to_insert in reversed(new_lines):
                    lines.insert(section_end_idx, line_to_insert)

            elif action.upper() == "DELETE":
                # 删除整个章节（包括标题行）
                del lines[target_idx : section_end_idx]
                
            else:
                logging.warning(f"操作 {i+1} 中未知的补丁类型 '{action}'。跳过。")
                
        except Exception as e:
            logging.error(f"应用补丁操作 {op} 时出错: {e}")

    # 将修改后的行列表重新组合成单个字符串
    return "".join(lines)