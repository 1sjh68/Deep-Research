# 智能 AI 内容创作框架 (Autonomous AI Content Generation Framework)

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

这是一个高度自主的 AI 内容生成与优化框架。它通过模拟一个由作者、审稿人、研究员和战略规划师组成的专家团队，围绕一个给定的复杂问题，自动地规划、生成、研究、修订并最终产出一篇结构完整、内容详实的高质量长篇文档。

## 核心特性 (Core Features)

- **大纲优先，迭代式生成 (Outline-First, Iterative Generation):** 项目首先由 AI 规划师生成一份详细的文档大纲，然后逐章逐节地进行内容创作，确保文章的结构性和逻辑性。
- **双 AI 协作：生成器与审稿人 (Dual AI Roles: Generator & Critic):** 系统内存在两个核心 AI 角色。“作者 AI” 负责内容创作，而 “审稿人 AI” 则负责从多个维度（准确性、完整性、逻辑性）对生成的内容进行严格评审，提出具体的改进意见。
- **基于工具调用的精确控制 (Tool-Based Control):** 框架大量使用 LLM 的工具调用（Function Calling）功能，将大纲生成、内容修订等任务转化为对结构化工具的调用。这确保了 AI 输出的 JSON 格式准确无误，极大地提高了系统的稳定性和可靠性。
- **RAG 增强的知识体系 (RAG-Enhanced Knowledge):**
  - **长期经验库:** 每次成功完成任务后，关键的产出（如最终文稿、成功的修订方案、研究简报）会被向量化并存入 ChromaDB 数据库，作为未来任务的参考经验。
  - **短期自反思 RAG:** 在单次任务中，已生成的文稿会被实时索引。这允许 AI 在进行修订时，对“正在写的”文档进行语义检索，从而做出更精准、更具上下文感知能力的修改。
- **自动化研究与知识空白填补 (Automated Research & Knowledge Gap Filling):**
  - “审稿人 AI” 能够智能识别出文稿中的“知识空白”（即需要外部信息支撑的论点）。
  - 系统会自动将这些空白转化为搜索引擎查询，通过 Google Search API 获取信息，并异步抓取和总结网页/PDF 内容，形成研究简报，反哺给“作者 AI”进行内容补充。
- **自适应规划 (Dynamic Planning):** 在完成一个主章节后，AI 规划师会重新评估剩余的大纲和已完成的工作，动态调整后续的写作计划，确保项目始终朝着最优路径前进。
- **高健壮性与可恢复性 (High Robustness & Resilience):**
  - 内置检查点（Checkpoint）机制，即使程序意外中断，也可以从上一个完成的步骤恢复任务。
  - 所有外部 API 调用都集成了基于 `tenacity` 的自动重试逻辑，能有效应对临时的网络或服务不稳定。

## 系统架构 (System Architecture)

本框架的工作流程模拟了一个真实的专家团队协作过程：

1.  **规划 (Plan):** AI 规划师根据用户问题，生成风格指南和文档大纲，并分配各章节的预估篇幅。
2.  **生成 (Generate):** 作者 AI 在 `ContextManager` 提供的精确上下文指导下，逐个章节地生成初始文稿。
3.  **评审 (Critique):** 审稿人 AI 对当前文稿进行全面分析，提供详细的反馈，并识别出知识空白。
4.  **研究 (Research):** AI 研究员根据知识空白，自动上网搜索、抓取、总结相关资料。
5.  **修订 (Patch):** 作者 AI 根据评审反馈和新的研究资料，生成一个结构化的“补丁”列表（而非重写全文），系统精确地将补丁应用到原文上。
6.  **迭代 (Iterate):** 重复步骤 3-5，直到达到预设的迭代次数，或审稿人 AI 对文稿质量感到满意。
7.  **定稿 (Finalize):** 进行最终的结论撰写和全文润色，输出最终文档。

## 安装与配置 (Installation & Setup)

#### 1. 先决条件
- Python 3.9 或更高版本
- Git

#### 2. 克隆项目
```bash
git clone <your-repository-url>
cd your_project_name
```

#### 3. 创建并激活虚拟环境
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

#### 4. 安装依赖
项目所有的依赖库都记录在 `requirements.txt` 文件中。
```bash
pip install -r requirements.txt
```

#### 5. 配置环境
这是最关键的一步。

1.  复制环境变量示例文件：
    ```bash
    cp .env.example .env
    ```
2.  编辑 `.env` 文件，填入你的个人信息和 API 密钥：

    ```dotenv
    # --- 核心 API 配置 ---
    # DeepSeek AI (必需)
    DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
    DEEPSEEK_BASE_URL="[https://api.deepseek.com/v1](https://api.deepseek.com/v1)"

    # 嵌入模型 API (推荐，用于 RAG 和经验库)
    EMBEDDING_API_KEY="your_embedding_api_key"
    EMBEDDING_API_BASE_URL="https://your_embedding_api_endpoint/v1"
    EMBEDDING_MODEL_NAME="bge-m3"

    # Google 搜索 API (推荐，用于研究功能)
    GOOGLE_API_KEY="your_google_api_key"
    GOOGLE_CSE_ID="your_google_programmable_search_engine_id"
    # GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account.json" # (可选，如果使用服务账户)

    # --- 任务配置 ---
    # 你想让 AI 解决的核心问题
    USER_PROBLEM="研究回旋镖飞行的具体原理，研究其运动如何取决于相关参数，最后要有可以联立求解运动轨迹的微分方程组，只要详细的理论推导，不许偏离主题，不允许讲工科部分，最后给我规范论文的格式，有引用，用中文回答我，要步步推导有逻辑，不确定的标注（此处不确定）"

    # 本地参考文件 (可选, 多个文件用逗号分隔)
    EXTERNAL_FILES="path/to/your/file1.pdf,path/to/your/document2.txt"

    # --- 运行参数 ---
    MAX_ITERATIONS=5       # 优化循环的最大次数
    INITIAL_SOLUTION_TARGET_CHARS=20000 # 期望的最终文档总字数
    INTERACTIVE_MODE=true # 是否开启交互模式 (在关键步骤会请求用户确认)
    ```

## 如何运行 (How to Use)

1.  **配置任务:** 在 `.env` 文件中，仔细设置 `USER_PROBLEM` 来定义你的写作任务。如果有本地参考资料，请配置 `EXTERNAL_FILES`。
2.  **启动脚本:** 运行 `main.py` 文件启动整个工作流。
    ```bash
    python main.py
    ```
3.  **查看产出:**
    - 所有的日志、中间产物、检查点和最终报告都会被保存在 `output/session_<timestamp>/` 目录下。
    - 最终的文稿通常命名为 `final_solution_v11_xxxxxx.txt`。

## 项目结构 (Project Structure)

```
your_project_name/
├── 📄 .env                  # 存储你的 API 密钥和环境变量
├── 📄 README.md             # 项目说明文档
├── 📄 requirements.txt        # 项目依赖库列表
├── 📄 main.py               # 程序的主入口
|
├── 📁 config/              # 配置模块
│   └── 📄 settings.py         # 存放 Config 类
|
├── 📁 core/                 # 核心业务逻辑模块
│   ├── 📄 context_manager.py  # 精确上下文管理器
│   └── 📄 patch_manager.py    # 内容补丁应用逻辑
|
├── 📁 planning/             # 规划与大纲模块
│   ├── 📄 outline.py          # 大纲生成与管理
│   └── 📄 tool_definitions.py # AI 工具的 JSON Schema 定义
|
├── 📁 services/             # 外部服务交互模块
│   ├── 📄 llm_interaction.py  # LLM API 调用封装
│   ├── 📄 vector_db.py        # 向量数据库与嵌入模型管理
│   └── 📄 web_research.py     # Google 搜索与网页抓取
|
├── 📁 utils/                # 通用工具函数模块
│   ├── 📄 file_handler.py     # 文件及检查点操作
│   └── 📄 text_processor.py   # 文本处理工具
|
└── 📁 workflows/            # 主要工作流编排模块
    └── 📄 generation.py       # 核心的迭代优化工作流
```

## 贡献指南 (Contributing)

欢迎对本项目进行贡献！你可以通过以下方式参与：
-   提交 Issue 来报告 Bug 或提出功能建议。
-   创建 Pull Request 来贡献你的代码。

在提交代码前，请确保你的代码风格与项目保持一致，并已在本地充分测试。

## 许可证 (License)

本项目采用 [MIT License](https://opensource.org/licenses/MIT) 授权。