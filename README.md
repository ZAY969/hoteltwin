# 无人酒店集群数字孪生 · 远程运营管控系统原型

数据全部由程序模拟,启动后持续运行。后端每秒推送一帧数据,前端驾驶舱实时刷新。

## 启动

```bash
pip install simpy pyyaml pandas numpy fastapi "uvicorn[standard]" matplotlib pytest
python server.py
```

然后在浏览器打开 <http://localhost:8000>。

驾驶舱上可以调节仿真速度、暂停、随机注入故障;点击任意客房可查看它的全部量化数据和曲线,并可对某个设备注入故障;客房图可按房态、室温、功率、CO2、门锁电量着色。

## 系统结构(与论文第 3 节的五层架构对应)

| 层 | 文件 | 内容 |
| --- | --- | --- |
| 物理层(模拟) | `hoteltwin/physical.py`、`hoteltwin/environment.py` | 虚拟酒店集群:预订、入住、退房、清洁、入住受阻、全店断网;每间客房的室温、湿度、CO2、功率、水流量、门锁电量、网络时延按物理模型连续演化并带传感器噪声;设备故障不发消息,只改变这些读数;接收修复、放行、催办、封房、解封指令 |
| 数据层 | `hoteltwin/engine.py` 的 `_publish` | 接收消息,更新孪生,转交决策层 |
| 孪生模型层 | `hoteltwin/states.py`、`hoteltwin/twin.py` | 客房状态机(6 状态 8 迁移)、遥测矩阵、电量与用水积分、推断的设备健康向量;客房单元、单店、集群三级汇总 |
| 对照方案 | `hoteltwin/traditional.py` | 传统驻店管理模式:客人报告、清洁发现、店员巡查;每店前台加工程人员处置 |
| 决策层 | `hoteltwin/decision.py` | 从遥测数据识别故障(阈值、持续时间、偏差趋势、心跳超时、自检、预测等规则)、优先级评分、三级处置分流(自动 → 远程人工 → 现场工单)、闭环验证、运营员队列、指标统计 |
| 应用层 | `server.py`、`frontend/index.html` | 后端接口与实时推送;运营驾驶舱页面 |

## 后端接口

| 接口 | 说明 |
| --- | --- |
| `GET /` | 驾驶舱页面 |
| `GET /api/state` | 当前完整状态(JSON) |
| `WS /ws` | 连接后先收到一帧完整状态,之后每秒一帧变化量;可发送 `{"speed":30}`、`{"paused":true}`、`{"inject":{"h":0,"r":5,"device":"lock"}}` |

## 其他脚本

| 命令 | 作用 |
| --- | --- |
| `python run_stage1.py` | 孪生模型一致性验证(30 天),结果在 `output/` |
| `python final_run.py` | 论文全部实验:对比、负荷扫描、敏感性分析,共 67 次运行,结果逐行写入 `output/final.jsonl` |
| `python final_agg.py` | 汇总上一步结果,生成 `output/final_agg.json` |
| `python det_table.py` | 分设备类别的检测性能(5 个种子) |
| `python selftest_run.py` | 主动自检实验,结果在 `output/selftest.json` |
| `python p23.py` | P2、P3 事件响应时限达标率核对 |
| `python figs.py`、`python figs2.py` | 生成论文图 1 至图 6,保存在 `figs/` |
| `python record_replay.py` | 录制一段数据,生成不需要后端也能播放的 `output/replay.html` |
| `python -m pytest tests -q` | 8 项自动检验 |

`experiments.py` 和 `supp.py` 是上述脚本调用的实验函数。Windows 下可直接双击 `start.bat` 启动驾驶舱。

全部量化指标的定义、单位、模型方程和检测规则见 `INDICATORS.md`。

## 论文结果文件

`output/final_agg.json`、`output/selftest.json`、`output/summary.json` 是论文表格所用的汇总结果,`figs/` 是论文插图。随机种子固定,重新运行上述脚本可复现。

## 参数

全部在 `config.yaml`。标注"文献"或"数据"的参数有公开出处(故障率:Lai 和 Yik 2012;入住率:华住集团 2025 年季度数据;退房清洁时长:Aguilar-Escobar 等 2021;CO2 产生率:Persily 和 de Jonge 2017;住宿晚数、每间人数、未到店率:Antonio 等 2019 数据集)。标注"设定"的参数没有公开出处,是本文的设定值,包括自动与远程处置成功率、处理时长、响应时限等,其影响由论文的敏感性分析检验。

## 范围说明

- 全部数据由程序模拟,没有接入真实酒店设备,结论限于仿真条件。
- 没有模拟消息丢失、传输延迟和传感器失效。
- 现场工单只按服从给定分布的时长处理,派发与接单机制不在本文范围内。
- 演示原型没有用户登录和数据库持久化。
