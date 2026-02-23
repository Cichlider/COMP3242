# SetX Agent Workflow Instructions (COMP3242 Adapted)

## 目标
将任意一周课程材料统一处理为三类产物：
- `raw`：逐页原始提取文本（不删减）
- `notes`：中文整理与可考知识点清单
- `exam`：英文试卷（问卷+答卷，优先 LaTeX）

本流程可复用于每周内容，不再要求文件名必须为 `setX.pdf`。

---

## 适用范围
- 课程：COMP3242/6242 Deep Learning
- 周目录：`wkYY/`（如 `wk01`, `wk02`）
- 讲义输入：`wkYY/*.pdf`（如 `01-introduction.pdf`, `02-mlps.pdf`, `lecture_notes.pdf`）
- 可选参考：`wkYY/labYY/*.ipynb`（用于补充与 lab 相关的可考点）

---

## 输入规范
每次处理一个 `LECTURE_ID`。若同一周有多个 PDF，分别处理或先合并知识点再出一套卷。

```text
WEEK_DIR   = wkYY
LECTURE_ID = 自定义标识（建议与文件名一致，如 01-introduction / 02-mlps）
PDF_PATH   = wkYY/<lecture-file>.pdf
LAB_PATH   = wkYY/labYY/labYY.ipynb (optional)
```

---

## 输出目录规范（固定）
在 `wkYY/LECTURE_ID/` 下创建：

```text
wkYY/
  <lecture-file>.pdf
  LECTURE_ID/
    raw/
      LECTURE_ID_full_extracted.txt
    notes/
      LECTURE_ID_中文完整复写.md
      LECTURE_ID_可考知识点清单.md
    exam/
      LECTURE_ID_exam_questions.tex
      LECTURE_ID_exam_solutions.tex
      LECTURE_ID_exam_questions.pdf
      LECTURE_ID_exam_solutions.pdf
```

若无 LaTeX 编译环境，保留 `.tex`，并额外输出：
- `LECTURE_ID_exam_questions.md`
- `LECTURE_ID_exam_solutions.md`

---

## 执行步骤（Agent 按序执行）

### 1) 建目录
- 创建：`wkYY/LECTURE_ID/raw`、`wkYY/LECTURE_ID/notes`、`wkYY/LECTURE_ID/exam`

### 2) PDF 逐页文本提取（raw）
- 必须逐页提取并保留页边界
- 输出：`wkYY/LECTURE_ID/raw/LECTURE_ID_full_extracted.txt`
- 要求：
  - 使用 `===== PAGE n =====` 分隔
  - 仅清理不可见破坏字符（如 `\0`）
  - 不删句、不改写、不重排

### 3) 中文完整复写（notes）
- 输出：`wkYY/LECTURE_ID/notes/LECTURE_ID_中文完整复写.md`
- 要求：
  - 采用模块化标题，不使用“第x页”标题
  - 忠实表达原义，允许同主题跨页合并
  - 图示页可简述，但与考试/lab相关信息不得省略

### 4) 可考知识点清单（notes）
- 输出：`wkYY/LECTURE_ID/notes/LECTURE_ID_可考知识点清单.md`
- 仅保留“可算、可证明、可编程实现、lab常用”的点：
  - 线性代数基础（向量/矩阵/张量、内积、范数、矩阵乘法、broadcast）
  - 机器学习基本框架（模型、损失、经验风险最小化、优化目标）
  - 神经网络核心（线性层、激活函数、MLP、参数量、前向/反向传播）
  - 自动微分与梯度（链式法则、计算图、梯度检查、常见梯度错误）
  - 训练策略（初始化、学习率、正则化、批大小、过拟合与泛化）
  - PyTorch 实操（tensor shape/dtype/device、autograd、DataLoader、训练循环）
  - 与当周 lab 直接关联的 API、数据流和调试点
- 不写课程行政/师资宣传/感想类内容

### 5) 生成英文试卷（exam）
- 输出问卷：`wkYY/LECTURE_ID/exam/LECTURE_ID_exam_questions.tex`
- 输出答卷：`wkYY/LECTURE_ID/exam/LECTURE_ID_exam_solutions.tex`
- 规则：
  - 语言必须为英文
  - 题型至少覆盖：conceptual + derivation/calculation + code reading/debugging
  - 涉及计算的知识点必须有计算/推导题
  - 涉及实现的知识点必须有 shape tracing 或 PyTorch 代码题
  - 答卷给出最终答案与关键步骤

### 6) 渲染 PDF（exam）
- 推荐命令：
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error <file>.tex`
- 若渲染失败：
  - 保留 `.tex`
  - 输出同内容 `.md` 版本作为兜底
  - 在最终回报中注明失败原因

### 7) 清理编译中间文件
- 仅保留：`.tex` 与 `.pdf`（或 `.md` 兜底文件）
- 推荐命令：
  - `latexmk -c LECTURE_ID_exam_questions.tex`
  - `latexmk -c LECTURE_ID_exam_solutions.tex`

---

## 质量验收清单（每次都要过）
- [ ] `raw/` 存在且含逐页文本
- [ ] `notes/` 两份 `.md` 都存在
- [ ] 中文复写为模块化结构，无“第x页”标题
- [ ] 知识点清单仅含可考/可算/可编程实现点
- [ ] `exam/` 含问卷+答卷（优先 `.tex` + `.pdf`）
- [ ] 计算题覆盖该讲义全部可计算知识点
- [ ] 至少一题考察 PyTorch shape/gradient/debugging
- [ ] `exam/` 无多余 LaTeX 中间文件

---

## 一键复用变量模板

```text
WEEK_DIR   = wkYY
LECTURE_ID = <lecture-id>
PDF_PATH   = wkYY/<lecture-file>.pdf
LAB_PATH   = wkYY/labYY/labYY.ipynb (optional)
```

文件命名自动套用：

```text
wkYY/LECTURE_ID/raw/LECTURE_ID_full_extracted.txt
wkYY/LECTURE_ID/notes/LECTURE_ID_中文完整复写.md
wkYY/LECTURE_ID/notes/LECTURE_ID_可考知识点清单.md
wkYY/LECTURE_ID/exam/LECTURE_ID_exam_questions.tex
wkYY/LECTURE_ID/exam/LECTURE_ID_exam_solutions.tex
wkYY/LECTURE_ID/exam/LECTURE_ID_exam_questions.pdf
wkYY/LECTURE_ID/exam/LECTURE_ID_exam_solutions.pdf
```

---

## 给 Agent 的标准任务指令（可直接复制）
请按 `SETX_AGENT_WORKFLOW (COMP3242 Adapted)` 执行 `wkYY/<lecture-file>.pdf` 全流程：
1. 建立 `raw/notes/exam` 目录结构
2. 提取逐页原文到 `raw`
3. 输出模块化中文完整复写到 `notes`
4. 输出“可考知识点清单”到 `notes`（结合当周 lab 可选）
5. 生成英文试卷问卷+答卷到 `exam`（含概念/计算/代码题）
6. 尝试渲染 PDF；失败则保留 tex 并输出 markdown 兜底
7. 最终回报所有输出路径、覆盖知识点与未完成项（如有）
