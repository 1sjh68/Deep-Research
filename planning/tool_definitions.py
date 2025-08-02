# planning/tool_definitions.py

"""
本模块定义了与 AI 交互时使用的“工具”的 JSON Schema。
这使得我们可以强制 AI 以结构化、可预测的格式返回数据，是系统稳定性的关键。
"""

def get_outline_review_tool_definition():
    """为大纲审查定义的专用工具 (Tool)"""
    # 定义“章节”的递归结构，以便 AI 能理解并生成包含子章节的嵌套计划
    chapter_schema = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "章节的标题，必须简洁明了。"
            },
            "description": {
                "type": "string",
                "description": "对本章节核心内容的2-3句话简要描述。"
            },
            "target_chars_ratio": {
                "type": "number",
                "description": "本章节预计占剩余总字数的比例（一个0到1之间的小数）。"
            },
            "sections": {
                "type": "array",
                "description": "子章节列表（可选），其结构与父章节相同。",
                "items": {"$ref": "#/definitions/chapter_definition"}
            }
        },
        "required": ["title", "description"] # 规定title和description是必填项
    }

    # 这是我们提供给AI的完整工具定义
    tools_definition = [
        {
            "type": "function",
            "function": {
                "name": "update_document_outline",
                "description": "根据评审意见和已完成的工作，更新或修正文档的剩余大纲。如果原始计划依然完美，则必须提交与原始计划完全相同的计划。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "revised_plan": {
                            "type": "array",
                            "description": "一个包含所有剩余章节对象的列表。",
                            "items": {"$ref": "#/definitions/chapter_definition"}
                        }
                    },
                    "required": ["revised_plan"],
                    "definitions": {
                        "chapter_definition": chapter_schema
                    }
                }
            }
        }
    ]
    return tools_definition

def get_initial_outline_tool_definition():
    """为初始大纲生成定义的工具"""
    # 定义“章节”模式，允许递归的子章节
    chapter_schema_for_initial_outline = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "本章节或小节的标题。"
            },
            "description": {
                "type": "string",
                "description": "对本章节内容和目的的简要描述（2-3句话）。"
            },
            "target_chars_ratio": {
                "type": "number",
                "description": "本章节预计占文档总长度的比例（例如，0.1代表10%）。顶层章节的比例总和应约为1.0。"
            },
            "sections": {
                "type": "array",
                "description": "可选的子章节或小节列表，每个都遵循此相同结构。",
                "items": {"$ref": "#/definitions/chapter_definition_initial"} # 递归定义
            }
        },
        "required": ["title", "description"] # target_chars_ratio 可以被AI初步省略，稍后会进行规范化
    }

    tools_definition = [
        {
            "type": "function",
            "function": {
                "name": "create_initial_document_outline",
                "description": "根据用户的问题陈述生成结构化的文档大纲。大纲应包括一个主标题和章节列表，每个章节都有标题、描述以及可选的子小节。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "整个文档/报告的主标题。"
                        },
                        "outline": {
                            "type": "array",
                            "description": "一个对象列表，其中每个对象代表文档的一个主要章节。",
                            "items": {"$ref": "#/definitions/chapter_definition_initial"}
                        }
                    },
                    "required": ["title", "outline"],
                    "definitions": {
                        "chapter_definition_initial": chapter_schema_for_initial_outline
                    }
                }
            }
        }
    ]
    return tools_definition

def get_patcher_tool_definition():
    """为内容修订（打补丁）定义的工具"""
    patch_action_enum = ["REPLACE", "INSERT_AFTER", "DELETE"]
    
    patch_object_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": patch_action_enum,
                "description": "要执行的补丁操作。"
            },
            "target_section": {
                "type": "string",
                "description": "目标小节的精确 Markdown 标题字符串（例如，'## 引言', '### 2.1 分析'）。这必须与文档中的某个标题完全匹配。"
            },
            "new_content": {
                "type": "string",
                "description": "该小节的新 Markdown 内容。对于 'REPLACE' 和 'INSERT_AFTER' 操作是必需的。对于 'DELETE' 操作应省略或为空字符串。"
            }
        },
        "required": ["action", "target_section"]
    }

    tools_definition = [
        {
            "type": "function",
            "function": {
                "name": "generate_json_patch_list",
                "description": "根据反馈生成一个 JSON 补丁操作列表，以修改 Markdown 文档。每个补丁通过其精确的标题来定位一个小节。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patches": {
                            "type": "array",
                            "description": "一个补丁操作列表。如果不需要任何更改，则应为空列表。",
                            "items": patch_object_schema
                        }
                    },
                    "required": ["patches"]
                }
            }
        }
    ]
    return tools_definition