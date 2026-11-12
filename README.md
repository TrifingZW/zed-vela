> [!IMPORTANT]
> Remove this line to confirm you've reviewed this PR before submitting.

# Zed Vela

Zed Vela 是 Zed 编辑器的一个分支，具有树图上下文管理，独立的 LLM 提供商面板等特性，并积极保持与官方上游同步.

## 特性

- 精简系统提示词
- 将所有 Border 改为 2px
- 更紧凑的工具栏
- 构建配置设为最优，同时编译时长最长
- 更改品牌为 Zed Vela
- 树图上下文管理
- 独立的 LLM 提供商配置面板
- 支持多个相同提供商
- 将编辑器的工具栏变得紧凑
- Agent 被中断或手动停止后可一键继续生成，无需重新输入提示词

## 行为

与官方 Zed 混用时，请留意以下数据归属：

- 用户目录未改名（Windows 下为 `%APPDATA%\Zed` 与 `%LOCALAPPDATA%\Zed`），settings.json、键位、线程数据库、扩展与语言服务器均与官方共用
- 单实例互斥锁与更新通道相互独立，两版可以同时运行，但会并发写同一批文件，以最后写入者为准
- 自动更新已关闭：官方更新源只提供上游 Zed 的安装包，更新需自行重新构建并安装 Zed Vela
- 安装包与资源管理器右键菜单使用独立的 AppId、Appx 标识与 COM CLSID，不会顶替官方 Zed 的安装记录或右键菜单
- 树状上下文仅 Vela 读写；官方版本保存同一线程时会丢弃分支结构，只保留当前分支
- 兼容提供商的 API Key 按提供商名称独立保存，与官方按 URL 保存的密钥互不覆盖；旧密钥首次加载时自动迁移，在 Vela 中重置则会连带清理旧条目
- 如需完全隔离，可用 `--user-data-dir` 指定独立的数据目录

## 开发

- [在 macOS 上构建](./docs/src/development/macos.md)
- [在 Linux 上构建](./docs/src/development/linux.md)
- [在 Windows 上构建](./docs/src/development/windows.md)

## 许可

Zed Vela 源自 Zed，源代码主要遵循 GPL-3.0-or-later，部分标注为 Apache-2.0 的组件以其标注为准。感谢 Zed Industries 与开源社区的每一份馈赠。
