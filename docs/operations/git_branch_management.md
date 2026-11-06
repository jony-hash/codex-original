# Git 分支治理与安全回档

本文件只定义稳定治理规则，不保存会快速过时的“当前分支”、远端数量、领先关系或
tag commit 表格。2026-08-12 的旧拓扑快照已归档到
[git branch topology snapshot](../archive/git_branch_topology_20260812.md)，不能替代现场核验。

## 只读核验

任何分支、回档、发布或证据 lineage 判断前，先运行：

```bash
git status --short
git branch --show-current
git branch -vv
git branch -r
git tag --sort=-creatordate
git log --oneline --decorate -20
```

需要判断提交关系时使用 `git merge-base`、`git branch --contains <commit>`、
`git show <ref>:<path>` 和 `git diff <base>...<head>`。文档中的 commit、branch 或 tag
只能作为带日期历史事实；实际 Git refs 始终是当前 source of truth。

## 稳定角色

- `main` 是历史骨架，不自动代表当前科学或运行主线。
- `rewrite/v2` 是 teacher-free identity v2 的稳定基线，不替换 v1 `train.py`。
- v1 历史中文分支和 v79 快照保存 LDM distillation 与诊断历史，不能签发 v2 evidence。
- `experiment/*` 与 `research/*` 是局部研究线；其存在不表示结果通过、允许 promotion
  或已发布到远端。
- `archive/*` tag 是只读里程碑。移动、删除或重签 archive tag 会改变历史引用，必须有
  明确用户授权和替代引用方案。

## 回档与比较规则

- 默认使用 `git show` 或 detached read-only inspection 查看历史，不覆盖当前工作树。
- 工作树有修改时，不运行 `reset --hard`、`checkout -- <path>` 或会丢失改动的清理命令。
- 回档前明确目标 ref、涉及路径、需要保留的用户修改和恢复方式；只回退请求范围内内容。
- `outputs/`、`data/raw/`、`.tmp/`、显式标记的 legacy namespaces 和未提交研究工件不随 Git ref 操作清理。
- 历史 evidence 只能由其原合同消费；切到旧 commit 不会自动恢复当前 acceptance 权限。

## 发布规则

发布前先执行 `git fetch --prune origin`，再核对目标 ref、上游关系、dirty state 和待推送
提交。不要根据本地 branch 名推断远端已经存在，也不要根据历史文档推断远端数量。
正式 v2 evidence 仍受 clean-ancestor、scoped Git diff、配置、split、checkpoint 和 artifact
内容绑定约束；分支发布本身不构成科学验收。

## 清理规则

- `git fetch --prune` 只清理本地对已删除远端 ref 的引用，可直接执行。
- 已合并的 `research/*`、`experiment/*` 分支只在用户明确授权后删除；不按日期或数量自动清理。
- 未合并分支、`archive/*` tag 和 `rewrite/v2` 基线不删除。
- 删除前记录分支名、最后 commit、是否已合并和对应证据路径。

## v1 与 v2 边界

v1 继续由 `train.py --stage` 驱动；v2 由 `python -m ldm_project.v2` 驱动。两者可共享
正式物理几何，但 checkpoint、训练输入和 evidence 不互换。diagnostic、research 和
archive refs 均不能绕过 current v2 的 fail-closed gate。
