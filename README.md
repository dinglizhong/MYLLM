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

---

## 🔬 项目详解

### 1. `train_llm`
**目标**：从零预训练与指令微调小型大语言模型  
**技术点**：
- 基于 MiniMind 数据集完成 Pretrain 与 SFT 完整流程
- 支持 `torchrun` 多卡分布式与 `DeepSpeed` ZeRO 优化
- 包含数据预处理、Tokenizer 训练、学习率调度、Checkpoint 保存等工程细节

### 2. `train_moe`
**目标**：实现混合专家（Mixture of Experts）架构的预训练与微调  
**技术点**：
- Top-k Gating 路由机制与负载均衡 Loss
- 稀疏激活降低计算量，提升模型容量
- 同样支持 `torchrun` / `DeepSpeed` 分布式训练

### 3. `train_multimodal`
**目标**：构建视觉-语言多模态理解模型  
**技术点**：
- 以 SigLIP 为视觉编码器，Qwen2.5 为语言底座
- 设计两层 MLP 作为 Projector，对齐图像与文本特征空间
- 冻结视觉与语言骨干，仅训练对齐层，降低训练成本
- 支持单图与多图输入场景

### 4. `train_siglip`
**目标**：从头训练 SigLIP 风格的图文对比学习模型  
**技术点**：
- ViT 视觉编码器 + 中文 RoBERTa 文本编码器
- 对比学习损失（Sigmoid-based Contrastive Loss）
- 使用 MUGE 中文图文数据集进行训练

### 5. `train_qwen3_next`
**目标**：复现下一代 LLM 混合架构（参考 Qwen3 设计方向）  
**技术点**：
- 标准 Softmax Attention 与 Gated DeltaNet 线性注意力交替使用
- 结合 MoE 与 Dense MLP 层，兼顾效果与效率
- 支持 KV Cache 与因果卷积，适配长文本推理
- 完整预训练 + SFT 流程

### 6. `knowledge_distillation_llm`
**目标**：大语言模型的知识蒸馏（离线 + 在线双范式）  
**技术点**：
- **离线蒸馏**：学生模型通过 KL 散度拟合教师模型的输出分布
- **在线策略蒸馏**：学生自生成样本，以与教师的 KL 散度作为奖励，通过 PPO 优化策略
- 具备从数据准备、训练到评估的完整脚本

### 7. `knowledge_distillation_embedding`
**目标**：Embedding 表示模型的蒸馏与轻量化  
**技术点**：
- 教师模型（Qwen3-Embedding-4B/8B）生成正负样本分数
- 学生模型通过 KL 散度学习相似度分布
- 支持 LoRA 高效微调与全参数微调两种模式

### 8. `knowledge_distillation_llm_cross_tokenizer`
**目标**：解决教师-学生模型使用不同 Tokenizer 时的蒸馏难题  
**技术点**：
- 词汇表对齐与映射策略
- 跨 Tokenizer 的 logits 转换与损失计算
- 支持 `torchrun` 分布式训练

### 9. `ppo`
**目标**：实现 PPO（Proximal Policy Optimization）RLHF 训练  
**技术点**：
- 完整四组件架构：Policy、Critic、Reward Model、Reference Model
- GAE（Generalized Advantage Estimation）计算优势函数
- 经验池复用、KL 散度约束、梯度裁剪等稳定训练技巧

### 10. `grpo`
**目标**：实现 GRPO（Group Relative Policy Optimization）强化学习算法  
**技术点**：
- 以组（Group）为单位采样，计算组内相对优势，降低对价值模型的依赖
- 支持 BF16 混合精度与梯度检查点
- 多奖励函数设计：正确性奖励、格式奖励、长度奖励等

### 11. `dapo`
**目标**：实现 DAPO（Decoupled Clip and Dynamic Sampling Policy Optimization）  
**技术点**：
- 在 GRPO 基础上解耦 Clip 范围，改进 Loss 计算方式（按 Batch 维度聚合）
- 支持多奖励函数加权与参考模型 KL 约束
- 更稳定的策略更新与更快的收敛速度

### 12. `deepseek_learn`
**目标**：深度拆解并复现 DeepSeek 系列核心技术  
**技术点**：
- **MLA（Multi-Head Latent Attention）**：低秩压缩 KV，降低显存占用
- **MTP（Multi-Token Prediction）**：单步多 token 预测，提升训练效率
- **GRPO**：DeepSeek 使用的强化学习训练流程
- **DSA（Delta Sequence Attention）**：差分序列注意力机制
- 全部基于 PyTorch 从零手写实现

### 13. `kimi_attnres`
**目标**：复现 Kimi 模型的 Attention Residual（注意力残差）机制  
**技术点**：
- 动态残差连接：对历史层输出加权求和，增强长距离依赖建模
- 融合标准 Attention 与 Gated DeltaNet 线性注意力
- 交替使用 MLP 与 MoE 层，构建高效稀疏架构
- 基于 `transformers` 自定义 `PreTrainedModel`

### 14. `s1`
**目标**：低成本复现 S1 强推理模型的训练流程  
**技术点**：
- 数据精选：从多源数学问题出发，通过 Gemini / DeepSeek-R1 生成推理轨迹
- 质量过滤 + 难度过滤（保留模型答错的难题）+ 推理链长度筛选
- 仅使用约 1000 条高质量数据微调 Qwen2.5-32B
- **测试时扩展（Test-Time Scaling）**：通过 Budget Forcing 强制模型延长思考，提升推理准确率

### 15. `RAG`
**目标**：构建融合向量检索与关键词检索的 RAG 系统  
**技术点**：
- **向量检索**：FAISS + sentence-transformers
- **关键词检索**：BM25 算法
- **结果融合**：RRF（Reciprocal Rank Fusion）综合排序
- 支持本地 LLM 与 OpenAI API 双后端生成答案

### 16. `deep_research`
**目标**：基于 MCP（Model Context Protocol）的自动化深度研究 Agent  
**技术点**：
- 使用 `FastMCP` 封装搜索工具
- 调用 Kimi API 完成：查询生成 → 网页相关性评估 → 上下文提取 → 迭代搜索
- 支持图片搜索与描述生成
- 依赖 `pymysql`、`requests`、`openai`、`pandas` 等工具链

### 17. `all_to_tool_call`
**目标**：提升大模型 API 的工具调用（Tool Calling）能力  
**技术点**：
- 区分直接工具调用（API 返回结构化字段）与间接工具调用（从 content 解析参数）
- 设计小模型解析大模型的调用决策，降低延迟与成本
- 基于 `transformers` 框架实现

### 18. `pdf2markdown`
**目标**：高精度 PDF 转 Markdown 解析工具  
**技术点**：
- 基于 `gptpdf` 二次开发
- 版面分析模型识别表格、图片、段落区域
- 多模态大模型生成结构化 Markdown 内容
- 适用于文档数字化与知识库构建场景

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
| Agent & 工具 | MCP, FastMCP, Web Search, Tool Calling, gptpdf |
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
