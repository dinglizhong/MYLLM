# MYLLM — 丁立中个人技术作品集

> 本仓库汇总了我在大语言模型（LLM）、视觉语言模型（VLM）及多模态领域的算法复现、训练实践与工程优化经验。所有项目均围绕开源小参数模型（如 Qwen2.5-0.5B/32B、MiniMind 等）展开，覆盖从预训练、微调到强化学习、知识蒸馏、Agent 应用的完整链路，可直接用于技术面试与能力展示。

---

## 📌 个人定位

- **方向**：大模型算法工程（预训练 / 微调 / RLHF / 蒸馏 / 多模态）
- **特长**：论文复现能力强，能从零实现前沿架构；熟悉训练全链路，具备工程落地经验
- **模型规模**：以开源小模型为主，降低实验成本的同时保证算法原理的完整还原

---

## 🗂️ 项目总览

| 序号 | 模块 | 核心能力 | 关键词 |
| :--- | :--- | :--- | :--- |
| 1 | [`train_llm`](#1-train_llm) | 从头预训练 + SFT 微调 | Pretrain, SFT, DeepSpeed, torchrun |
| 2 | [`train_moe`](#2-train_moe) | 混合专家（MoE）架构训练 | MoE, Top-k Gating, 稀疏激活 |
| 3 | [`train_multimodal`](#3-train_multimodal) | 视觉-语言多模态对齐 | VLM, SigLIP, Qwen2.5, MLP Projector |
| 4 | [`train_siglip`](#4-train_siglip) | 从零训练 SigLIP 图文对齐模型 | Contrastive Learning, ViT, RoBERTa |
| 5 | [`train_qwen3_next`](#5-train_qwen3_next) | 下一代混合注意力架构预训练 | Gated DeltaNet, MoE, KV Cache |
| 6 | [`knowledge_distillation_llm`](#6-knowledge_distillation_llm) | 大模型知识蒸馏（离线 + 在线） | KD, SFT Distill, Policy Distill, PPO |
| 7 | [`knowledge_distillation_embedding`](#7-knowledge_distillation_embedding) | Embedding 模型知识蒸馏 | sentence-transformers, LoRA, KL 散度 |
| 8 | [`knowledge_distillation_llm_cross_tokenizer`](#8-knowledge_distillation_llm_cross_tokenizer) | 跨 Tokenizer 的 LLM 蒸馏 | Cross-Tokenizer, Vocabulary Align |
| 9 | [`ppo`](#9-ppo) | PPO 强化学习人类反馈对齐 | RLHF, GAE, Critic, Reward Model |
| 10 | [`grpo`](#10-grpo) | GRPO 组相对策略优化 | Group Relative Advantage, BF16 |
| 11 | [`dapo`](#11-dapo) | DAPO 动态采样策略优化 | Decoupled Clip, Dynamic Sampling |
| 12 | [`deepseek_learn`](#12-deepseek_learn) | DeepSeek 关键技术拆解与复现 | MLA, MTP, GRPO, DSA |
| 13 | [`kimi_attnres`](#13-kimi_attnres) | Kimi Attention Residual 机制复现 | Attention Residual, MoE, Gated DeltaNet |
| 14 | [`s1`](#14-s1) | 低成本强推理模型训练复现 | Test-Time Scaling, Budget Forcing, Data Curation |
| 15 | [`RAG`](#15-rag) | 检索增强生成系统 | FAISS, BM25, RRF, LangChain |
| 16 | [`deep_research`](#16-deep_research) | MCP 深度研究 Agent | Web Search, Relevance Scoring, Iterative Research |
| 17 | [`all_to_tool_call`](#17-all_to_tool_call) | LLM 工具调用能力增强 | Tool Calling, Function Call, Parameter Parsing |
| 18 | [`pdf2markdown`](#18-pdf2markdown) | 多模态 PDF 转 Markdown | Layout Analysis, Vision-Language, gptpdf |
| 19 | [`langgraph_agent`](#19-langgraph_agent) | 基于 LangGraph 的任务规划与执行 Agent | LangGraph, Planner, Agent, Workflow |

---

## 🔬 项目详解

### 1. `train_llm`
**目标**：从零预训练、指令微调与人类偏好对齐（DPO）的小型大语言模型  
**技术点**：
- **数据与分词器**：使用 `train_tokenizer.ipynb` 训练自定义 BPE 词表，`dataset.py` 构建高效的流式或加载数据流
- **预训练与SFT**：基于 MiniMind 数据集完成 `train.py` (Pretrain) 与 `sft_train.py` (SFT) 完整流程
- **直接偏好优化**：增加 `dpo_train.py` 脚本，基于 DPO (Direct Preference Optimization) 实现人类偏好对齐，取代传统 PPO
- **分布式计算**：支持 `torchrun` 多卡分布式与 `DeepSpeed` ZeRO 优化，包含学习率调度、混合精度训练与容错 Checkpoint 恢复

### 2. `train_moe`
**目标**：实现混合专家（Mixture of Experts）架构的预训练与微调，支持 Dense 结构转化为 MoE  
**技术点**：
- **路由机制**：在 `moe_train.py` 中实现了 Top-k Gating 路由机制，并引入负载均衡损失（Load Balancing Loss）防止专家坍塌
- **稀疏激活**：动态稀疏激活降低计算量，在相近的总参数量下大幅提升模型有效容量
- **数据工程**：利用统一的 `sft.jsonl` 与 `train.jsonl`，实现了完整的模型流转与多卡 `DeepSpeed` 微调与推理 (`moe_test.py`)

### 3. `train_multimodal`
**目标**：构建并训练视觉-语言多模态理解模型（VLM）  
**技术点**：
- **架构组装**：以 SigLIP（支持 `siglip-base-patch16-224` 与 `siglip-so400m-patch14-384`）为视觉编码器，Qwen2.5 为语言底座
- **跨模态对齐**：搭建并训练两层 MLP 作为 Projector，对齐图像与文本特征空间 (`test.py`)
- **多图上下文**：在 `sft_train_multi_images.py` 中扩展模型支持多图像交叉注意力与长上下文图片理解
- **训练策略**：冻结视觉与语言骨干，仅训练对齐层，显著降低训练成本

### 4. `train_siglip`
**目标**：从头训练 SigLIP 风格的图文对比学习模型  
**技术点**：
- **双塔编码器**：ViT 视觉编码器 + 中文 RoBERTa 文本编码器 (`model.py`)
- **对比学习设计**：复现 Sigmoid-based Contrastive Loss，消除对全局 Softmax 归一化的依赖，更利于 Batch 并行
- **语料处理**：编写 `data_process.ipynb` 和 `dataset.py` 利用 MUGE 中文图文数据集完成端到端收敛训练

### 5. `train_qwen3_next`
**目标**：复现下一代 LLM 混合架构（参考类似 Qwen3 混合模型的演进方向）  
**技术点**：
- **混合注意力**：标准 Softmax Attention（捕获精确局部特征）与 Gated DeltaNet 线性注意力（兼顾长距离与速度）交替使用
- **MoE + Dense**：结合 MoE 与 Dense MLP 层，构建出效率与效果兼备的稀疏网络
- **长文本推理**：支持自回归 KV Cache 管理与因果卷积，包含 `pretrain.py`, `sft_train.py` 及独立的 `moe_test.py` 验证管线

### 6. `knowledge_distillation_llm`
**目标**：大语言模型的知识蒸馏实践（离线日志与在线策略双范式）  
**技术点**：
- **数据生成**：通过 `build_new_dataset.py` 调用 OpenAI/本地更强模型生成响应与 Logits，利用 `check_dataset.py` 进行质量筛查
- **离线 SFT 蒸馏**：`train_sft_lora.py` 通过 KL 散度（KL Divergence）拟合教师模型的输出分布，兼顾硬标签与软标签损失
- **在线强化蒸馏**：`on_policy_distillation_train_rl.py` 实现学生模型自生成样本，以与大型教师模型的输出散度为 Reward，利用 PPO 优化策略

### 7. `knowledge_distillation_embedding`
**目标**：Embedding 文本表示模型的蒸馏与轻量化  
**技术点**：
- **数据构建**：`get_distillation_data_local.py` 与 `get_distillation_data_openai.py` 实现教师模型（Qwen3-Embedding 等）打分，构建包含难负例的样本对
- **损失优化**：模型通过 KL 散度与 MSE 损失学习教师的稠密相似度距离分布
- **测试评估**：包含数据汇总 `merge.py` 和全面的评估测算脚本 `evaluation.py`，并支持 LoRA / 全参数蒸馏模式

### 8. `knowledge_distillation_llm_cross_tokenizer`
**目标**：解决教师与学生模型词表（Tokenizer）不一致时的蒸馏技术难题  
**技术点**：
- **词汇表映射**：自动寻找 Token 交叉匹配映射，解决空间不对齐的 Logits 转移
- **交叉分布计算**：实现跨 Tokenizer 的概率分布映射与损失加权计算，克服了直接对应 Vocabulary Index 失败的瓶颈
- **分布式支撑**：基于 `utils.py` 和 `dataset.py` 构建并跑通了整个跨词表 `torchrun` 分布式蒸馏管线

### 9. `ppo`
**目标**：实现从零开始的 PPO（Proximal Policy Optimization）RLHF 强化学习人类反馈对齐  
**技术点**：
- **四模型协同**：完整构建 Actor Policy、Critic Value Model、Reward Model 和 Reference Model，并协调四者内存加载
- **GAE 估算**：使用 GAE（Generalized Advantage Estimation）精准计算时序优势函数，稳定梯度方差
- **训练收敛技巧**：在 `ppo_train.py` 中利用经验池复用、PPO Clip、KL 散度动态惩罚约束以及梯度裁剪等策略避免大模型策略崩塌

### 10. `grpo`
**目标**：实现 GRPO（Group Relative Policy Optimization）强化学习算法，摆脱大体积 Critic 模型  
**技术点**：
- **单模型优势估计**：以组（Group）为单位（如同一 Prompt 采样 N 次），在 `train.py` 中直接计算组内相对优势（Z-score 归一化）
- **多维 Reward 设计**：`reward_func.py` 实现多维度规则与打分加权：如正确性奖励、XML 格式扣分、思考长度奖励等
- **轻量与显存优化**：去掉了独立 Critic 模型设计，结合 BF16 混合精度与梯度检查点，使超小显存跑 RL 成为可能

### 11. `dapo`
**目标**：实现 DAPO（Decoupled Clip and Dynamic Sampling Policy Optimization）前沿策略优化  
**技术点**：
- **解耦优化**：在 GRPO 的基础上进一步解耦 Clip 范围规则，引入动态采样范围控制
- **Loss 聚合与截断**：改变常规 Token Level KL Penalty 和 Loss 计算方式，在 `train.py` 按 Batch/Group 维度更好地应对长短序列差异
- **评估体系**：利用 `reward_func.py` 特殊指标设计提供更稳定的策略更新与更快的模型收敛速度

### 12. `deepseek_learn`
**目标**：底层解构并用 PyTorch 纯手工复现 DeepSeek 系列论文中的几大杀器架构  
**技术点**：
- **MLA (Multi-Head Latent Attention)**：在 `MLA.py` 复现将 KV Cache 利用低秩压缩的技术方案并融合 RoPE 旋转位置编码，极大降低推理显存
- **MTP (Multi-Token Prediction)**：在核心文件中实现额外预测头的追加方案，一步预测多个未来 Token 增加数据利用率
- **DSA (Delta Sequence Attention)** / **mHC**：研究与拆解了深层特征抽象方案，包含 `engram.ipynb` 和 `mHC.ipynb` 的学习笔记与可视化

### 13. `kimi_attnres`
**目标**：复现 Kimi 模型的 Attention Residual（注意力残差）网络机制  
**技术点**：
- **长上下文记忆增强**：动态残差连接对历史层输出加权求和，克服深度加深时的远期 Token 信息衰减
- **多模块复合**：融合并打通了 标准 Attention、Gated DeltaNet 线性注意力，并交替使用 MLP 与 MoE 稀疏层
- **完整生态**：利用自定义 `PreTrainedModel` 构建了 `sft_train.py` 微调与分词处理，兼容 HuggingFace 社区加载标准

### 14. `s1`
**目标**：低算力低成本复现 S1 深度逻辑思考与强推理模型的训练与数据工程路线  
**技术点**：
- **数据精炼**：结合 Gemini / DeepSeek-R1 生成的多源数学推理轨迹，进行极其严格的质量、难度与长思维链筛选，仅提纯不到两千条核心集
- **强化 SFT**：在 `s1_train.py` 微调 Qwen2.5 等基座，引导模型进入类似 `<think>` `</think>` 的逻辑循环闭环
- **Test-Time Scaling 策略**：集成测试时扩展与 Budget Forcing 技术，在生成时强行拉长思考探索节点以换取准确性的非线性上涨

### 15. `RAG`
**目标**：针对垂直领域（医疗文本等）融合多路召回与重排技术的检索增强生成系统（RAG）  
**技术点**：
- **双路检索召回**：向量检索层面使用 FAISS 引入重计算和 sentence-transformers 建模语义；关键词层面打通 BM25 弥补精准词匹配的不足 (`rag.ipynb`)
- **知识重排融合**：RRF (Reciprocal Rank Fusion) 核心算法计算倒数综合排序，从多种分发池提取统一的最优文档块
- **后端双通道**：利用 LangChain 串联了本地自测开源 LLM 以及云端 OpenAI API 处理文本切分 (`medical_data.txt`) 并回答问题

### 16. `deep_research`
**目标**：基于新一代模型上下文协议（MCP）规范设计的全自动互联网深度调研 Agent  
**技术点**：
- **MCP 构建**：使用 `search_mcp.py` 和 FastMCP 范式封装联网搜刮工具，并对外提供标准接口
- **Agent 状态机循环**：主控制代码 `client.py` 调用具有长上下文的 Kimi/DeepSeek 核心模型完成：生成搜寻语句 → 调用搜寻 → 解析网页内容 → 追加思考迭代的重度作业
- **系统层整合**：结合 SearXNG/Duckduckgo 搜索组件与 Prompt 定制 (`prompts.py`) 构建了不设预设轮次的探索者闭环模型

### 17. `all_to_tool_call`
**目标**：让不具备原生 Function Calling 的轻量模型获得高质量 API 工具调度能力的训练方案  
**技术点**：
- **Schema 结构树解析**：`all_to_tool_call.py` 中制定了特制的系统提示词，将复杂工具参数定义注入并转义至标准文本输入
- **正则化输出控制**：将输出重组为类似 JSON 解析的统一规范格式（直接调用/间接调用）并拦截失效格式
- **效率下探分析**：专门测试与改进了参数容错解析逻辑 (`test.ipynb`)，显著降低 API Agent 二次封装时的响应开销

### 18. `pdf2markdown`
**目标**：为 RAG 和大语料集构建提供针对非结构化多模态文档高保真解析的 PDF 提取工具  
**技术点**：
- **OCR 与布局分析**：基于开源版 `gptpdf` 方案并二次开发 `pdf2markdown.py`，内嵌版面感知与切割坐标的精准定位（表格、页眉页脚、浮动图片）
- **多模态恢复**：调集 VLM 进行片段图表精析并转换为标准 Markdown Table 以及图片锚点引用
- **全自动化**：非常适合在庞大的文档数据工厂中扮演知识进入大模型训练管线前的数据摄入枢纽

### 19. `langgraph_agent`
**目标**：基于 LangGraph 的高度可定制任务规划与自动化工作流 Agent 引擎  
**技术点**：
- **流编排**：利用 LangGraph 中的 `StateGraph` 构建核心任务流图（见 `graph.py` 与 Mermaid 配置），包括 `Planner` -> `Execute` -> `Update` -> `Report`
- **状态维护**：以 `state.py` 全局追踪历史图调度与环境反馈，在复杂环境状态下决定分支和终止循环（Condition Edges）
- **表格分析演示**：`tools.py` 内部实现了 CSV 和结构数据读取与命令行运行反馈环境，结合 `student_habits_performance.csv` 提供了一个自动化分析与建模任务的代码实例

---

## 🛠️ 技术栈

| 类别 | 技术/框架 |
| :--- | :--- |
| 深度学习框架 | PyTorch, Transformers, DeepSpeed |
| 多模态 | PIL, SigLIP, ViT, RoBERTa, Vision-Language Projector |
| 检索与表示 | FAISS, sentence-transformers, BM25, LangChain |
| RLHF & 强化学习 | PPO, GRPO, DAPO, GAE, Reward Modeling |
| 知识蒸馏 | KL Distillation, Policy Distillation, LoRA, Cross-Tokenizer Align |
| 数据工程 | pandas, datasets, JSON/JSONL 处理, 数据清洗与筛选 |
| Agent & 工具 | MCP, FastMCP, Web Search, Tool Calling, gptpdf, LangGraph |
| 分布式训练 | torchrun, DeepSpeed ZeRO, BF16, Gradient Checkpointing |

---

## 💡 亮点总结

1. **全链路覆盖**：从数据清洗、Tokenizer 训练、Pretrain、SFT、RLHF、蒸馏到 Agent 应用，完整掌握 LLM 生命周期。
2. **前沿复现能力**：独立复现 DeepSeek（MLA / MTP / GRPO）、Kimi（Attention Residual）、S1（Test-Time Scaling）等热点技术。
3. **强化学习深度**：不仅实现标准 PPO，还深入 GRPO、DAPO 等改进算法，具备多奖励函数设计与策略优化经验。
4. **蒸馏创新**：覆盖 Embedding 蒸馏、跨 Tokenizer 蒸馏、在线策略蒸馏等多种范式，适配不同落地场景。
5. **工程可落地**：所有项目均提供可运行的训练脚本，支持分布式训练与混合精度，注重代码质量与可复现性。

---

## 📬 联系方式

- **姓名**：丁立中
- **GitHub**：您可在此仓库中查看完整代码与提交记录
- **求职意向**：大模型算法工程师 / LLM/VLM 算法工程师 / 预训练与对齐方向

> 欢迎 HR 与技术负责人查阅代码细节，如需进一步演示或技术交流，可随时联系！
