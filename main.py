# main.py

import os
import sys
import logging
import asyncio
import nest_asyncio
from datetime import datetime

# --- 路径修正代码 ---
# 确保项目根目录在 Python 搜索路径中，以便正确导入所有模块
# 这应该在所有自定义模块导入之前执行
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    # 在某些交互式环境（如某些版本的 notebooks）中 __file__ 未定义
    # 此时我们假设工作目录就是项目根目录
    project_root = os.getcwd()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
# --- 路径修正结束 ---

# 从重构后的模块中导入必要的类和函数
from config import Config
from services.vector_db import EmbeddingModel, VectorDBManager
from workflows.generation import generate_extended_content_workflow


async def main():
    """
    项目的主异步执行函数。
    负责初始化配置、服务，并启动内容生成工作流。
    """
    # 1. 初始化配置类
    # Config 类会自动尝试从 .env 文件加载环境变量
    config = Config()

    # 2. 设置日志系统
    # 日志会同时输出到控制台和 output/session_<timestamp>/session.log 文件
    config.setup_logging()
    logging.info("--- 智能内容创作框架启动 ---")

    # 3. 初始化核心服务
    try:
        # 初始化与 LLM 的连接客户端
        config._initialize_deepseek_client()
    except Exception as e:
        logging.critical(f"致命错误：无法初始化 DeepSeek 客户端: {e}. 程序即将退出。")
        sys.exit(1)

    # 初始化嵌入模型和向量数据库管理器
    # 即使这些服务初始化失败，主程序仍可继续，只是会缺失 RAG 相关功能
    vector_db_manager_instance = None
    try:
        embedding_model_instance = EmbeddingModel(config)
        if embedding_model_instance and embedding_model_instance.client:
            vector_db_manager_instance = VectorDBManager(config, embedding_model_instance)
        else:
            logging.warning("嵌入模型客户端初始化失败。RAG 和长期记忆功能将被禁用。")
    except Exception as e:
        logging.error(f"初始化嵌入或向量数据库管理器时出错: {e}。功能将受限。", exc_info=True)

    # 4. 从环境变量加载任务具体信息
    config.user_problem = os.getenv("USER_PROBLEM", "请详细阐述一下人工智能的未来发展趋势。")
    external_files_str = os.getenv("EXTERNAL_FILES", "")
    config.external_data_files = [p.strip() for p in external_files_str.split(',') if p.strip() and os.path.exists(p.strip())]

    # 5. 打印核心运行参数，启动工作流
    logging.info(f"任务问题 (前100字符): {config.user_problem[:100]}...")
    logging.info(f"外部参考文件: {config.external_data_files if config.external_data_files else '无'}")
    logging.info(f"最大迭代次数: {config.max_iterations}")
    logging.info(f"交互模式: {config.interactive_mode}")

    final_result = "错误：由于未知原因，工作流未能成功运行。"
    try:
        # 6. 调用并等待核心工作流完成
        final_result = await generate_extended_content_workflow(
            config, 
            vector_db_manager_instance
        )
    except Exception as e:
        logging.critical(f"主工作流程 'generate_extended_content_workflow' 发生未捕获的严重异常: {e}", exc_info=True)
        final_result = f"错误：工作流程因严重失败而终止。详情请查看日志。 {e}"

    # 7. 输出并保存最终结果
    logging.info("\n--- 主脚本执行完毕 ---")
    if final_result and not final_result.startswith("错误："):
        logging.info(f"成功生成内容。最终文档长度: {len(final_result)} 字符。")
        
        # --- 添加的代码：将最终结果保存到文件 ---
        try:
            output_filename = f"final_solution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            # 确保会话目录存在
            if config.session_dir and os.path.isdir(config.session_dir):
                output_filepath = os.path.join(config.session_dir, output_filename)
                with open(output_filepath, "w", encoding="utf-8") as f:
                    f.write(final_result)
                logging.info(f"🎉 最终报告已成功保存至: {output_filepath}")
            else:
                logging.error("会话目录不存在，无法保存最终文件。")

        except Exception as e:
            logging.error(f"保存最终报告时发生错误: {e}")
        # --- 添加结束 ---

    else:
        logging.error(f"脚本执行结束，但最终结果是错误或为空: {final_result}")


if __name__ == "__main__":
    # 应用补丁，以支持在某些环境（如 Jupyter Notebook）中嵌套执行 asyncio 事件循环
    nest_asyncio.apply()
    
    try:
        # 运行主异步函数
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序被用户手动中断。")
    except Exception as e:
        # 捕获任何在 main() 之外或 asyncio.run 期间发生的顶层错误
        logging.basicConfig(level=logging.INFO) # 确保即使日志设置失败也能打印
        logging.critical(f"在启动或运行主异步任务时发生致命错误: {e}", exc_info=True)