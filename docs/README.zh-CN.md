# agent-snapshot-resume(中文说明)

用途:给一条已经跑完的 agent 轨迹(比如失败的那条)和一个步号 k,在沙箱里让 agent 从第 k 步接着跑。接着跑时,工作区文件和模型看到的上下文都和原来第 k 步时一样。

要同时恢复两样东西:

1. **环境**:在源沙箱里按顺序重放第 1 到 k-1 步的副作用(写文件、执行命令、后台任务),检查通过后拍快照。每次续跑都从这个快照克隆一个新沙箱。
2. **会话**:把 agent 自己的会话文件截断到第 k 步之前,用 agent 自带的续跑命令接着跑(Claude Code 用 `claude --resume <sid> --print`)。沙箱里有一个代理,会改写每一次模型请求:删掉续跑时额外加进去的内容,把日期固定成原来那天,需要时再加一条提示。

现在只支持 Claude Code。和具体 agent 有关的代码都放在 `agents/claude_code.py` 一个模块里,要支持 Codex 等其他 agent,再写一个同样结构的模块就行,做法见 [adding-an-agent.md](adding-an-agent.md)。

沙箱需要兼容 E2B 接口并且支持快照,目前在阿里云函数计算(FC)沙箱上验证过。作业用 Harbor 0.23.0 运行(版本已固定)。

## 为什么要快照和会话一起恢复

- 从第 1 步重新跑:要再花一遍模型调用,而且模型可能还没到第 k 步就走了另一条路。
- 只恢复会话:工作区是空的,agent 之前建的文件都不在。
- 只恢复工作区再发一条新提示:模型看到的上下文变了。

所以两样都恢复,并且都要检查:会话用 `equiv` 检查,环境用指纹检查。每个 (轨迹, k) 只做一次快照,之后每次克隆几秒钟就能起来,文件和源沙箱一致。

## 流程

| 步骤 | 命令 | 在哪里跑 | 是否调用模型 | 涉及的密钥 |
|---|---|---|---|---|
| 生成 bundle | `asr build-bundle` | 本机,不联网 | 否 | 无 |
| 注册模板 | `asr build-template` | 沙箱服务 | 否 | 私有镜像仓库账号(按需) |
| 回放并拍快照 | `asr prepare-snapshot` | 一个源沙箱 | 否 | 任何密钥都不进沙箱 |
| 续跑 | `asr launch`(调用 Harbor) | 每个 trial 一个克隆沙箱,另加 verifier 沙箱 | 是(`--fake` 时否) | key 只存在于代理进程的内存里 |

## 快速上手

```bash
uv sync
cp .env.example .env          # 填 E2B_*、ASR_UPSTREAM_URL、ASR_UPSTREAM_KEY
uv run pytest -q              # 离线测试,不用沙箱,也不调模型

# 0. 先用仓库自带的合成样例试一下离线部分
uv run asr build-bundle --trial-dir examples/fixtures/synthetic-trial \
    --task-dir examples/fixtures/sum-task --step 4 --out bundles
uv run asr selfcheck --bundle bundles/synthetic-trial-syntheti-s4-same     # 应输出 equivalent: true

# 1. 用真实 trial 生成 bundle(--step 填要重新生成的第一个 ATIF step_id)
uv run asr build-bundle --trial-dir <trial 目录> --task-dir <task 目录> --step 27 --out bundles
# 2. 用任务镜像注册一个支持快照的模板
uv run asr build-template --task-dir <task 目录>
# 3. 在源沙箱里回放 k 之前的步骤并拍快照
uv run asr prepare-snapshot --task-dir <task 目录> --bundle bundles/<run_id> \
    --template state/template-<task>.json --tag <task>-s27
# 4. 干跑:不调模型,只验证上下文等价
uv run asr launch --bundle bundles/<run_id> --task-dir <task 目录> \
    --snapshot-state state/snapshot-<task>-s27.json --fake --no-verifier
uv run asr equiv <原 session.jsonl> <cut_line> <job 目录>/.../first_request_rewritten.json --sentinel <sentinel>
# 5. 正式续跑(加 --dry-run 只生成并打印 job.json,不启动 Harbor)
uv run asr launch --bundle bundles/<run_id> --task-dir <task 目录> --snapshot-state state/snapshot-<task>-s27.json
# 6. 清理
uv run asr sandboxes; uv run asr sandboxes --kill-id <id>; uv run asr delete-snapshot <snapshot_id>
```

`launch` 建议加 `--original-trial-dir <原 trial 目录>`，让续跑的 agent 沿用原始运行中不含密钥的 env 和 kwargs。原配置里 `${VAR}` 形式的值（例如 `CLAUDE_CODE_MAX_OUTPUT_TOKENS`）会从当前 shell 读取，读不到就丢弃并给出警告。这时要用 `--agent-env VAR=值` 补上原值，否则 `max_tokens` 这类请求参数会和原始运行不同。`equiv` 只检查消息前缀，参数是否一致要另外对比整条首请求。

选 k 时看 `bundle.json` 里的 `resume_cost`:`bg_inflight` 和 `subagents_inflight` 最好都是 0,也就是第 k 步时没有还在跑的后台任务和子 agent。

## 配置项

| 变量 | 含义 |
|---|---|
| `E2B_API_KEY` / `E2B_API_URL` / `E2B_DOMAIN` | 沙箱服务的凭据和地址。FC 的 key 按地域绑定,地址要和 key 在同一个地域 |
| `ASR_UPSTREAM_URL` | 模型服务地址,要兼容 Anthropic `/v1/messages`,并且沙箱里能访问到 |
| `ASR_UPSTREAM_KEY` | 模型 key。只从运行 Harbor 的进程环境里读,不会写进 job 配置、bundle、快照或任何文件 |
| `ASR_MODEL` | 续跑时用的模型名,不填就用原轨迹的模型 |
| `ASR_OWNER_TAG` | 写进沙箱 metadata 的 owner,也是模板名前缀,默认 `asr`。`asr sandboxes` 只列出和删除带这个标签的沙箱 |
| `REGISTRY_USERNAME` / `REGISTRY_PASSWORD` | 任务镜像在私有仓库时才需要 |

## "等价" 指什么

`asr equiv` 比较续跑后发出的第一条请求和原会话前缀 `lines[:cut]`。原模型看到过的每个内容块都必须按原顺序、逐字节一样地出现在请求里,包括指令、assistant 文本、thinking 及其签名、tool_use 和 tool_result。请求里多出来的内容只能是以下几种:固定成原日期的日期提醒,由前缀里的 attachment 渲染出来的系统消息,以及 hint 模式下的唯一一条提示。续跑时插入的 `Continue from where you left off.`、合成的 `No response requested.` 和 sentinel 一旦残留,就判为不等价。

**这个检查不覆盖以下几项,它们可能和原轨迹不同**:system prompt 里的环境信息(内核版本、`Shell:` 那行),模型服务的线路,以及采样结果。等价只针对输入上下文,续跑生成的内容本来就会不一样。

## 已知缺口

- **shell 当前目录不恢复**:Claude Code 续跑后 Bash 从项目根目录开始。如果原来已经 `cd` 到别的目录,续跑后第一条用相对路径的命令可能失败,模型一般下一步会自己改过来。尽量选当前目录就是项目根目录的 k。
- **隐藏状态不恢复**:第 k 步时还在运行的进程、没有记录的进程写出的文件、随机种子都恢复不了。
- **回放结果和原来不一致时,不拍快照**:先看回放报告。如果差异确实无害,在 `bundle.json` 里加一条和命令 sha256 绑定的豁免。
- 会话里的数值如果被替换成了 `[REDACTED]`,前缀就不是原样了,这种轨迹不要拿来续跑。
- Claude Code 固定用 2.1.207 版(二进制的 sha256 也固定)。换版本会改变工具列表或 system prompt。
- 离线测试没有覆盖真实沙箱这一段(建模板、准备源沙箱、克隆、真实调用模型)。`tests/test_recovery_live.py` 是一个需要手动开启的小型线上测试(设 `ASR_LIVE_TESTS=1`)。

## 安全

- 沙箱里的 agent 只拿到一个假 key(`dummy-resume`),请求先发到本机的续跑代理(4031 端口)。
- 真 key 只在执行一次命令时作为环境变量传进沙箱,随后通过管道交给直通代理(4013 端口),之后只存在于这个代理进程的内存里。它不落盘,不出现在命令行里,也不在进程环境变量里。`tests/test_proxy_credentials.py` 会实际运行生成的启动脚本来检查这些。
- 拍快照时沙箱里没有任何 key,所以快照里也没有。

更多设计细节见 [design.md](design.md)(英文)。
