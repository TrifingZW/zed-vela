# Zed Vela

Zed Vela 是 Zed 编辑器的一个分支。它不另起炉灶，而是站在 Zed 的肩膀上，以变基的方式紧随官方上游，把每一次同步都当作一次吸收，把每一处定制都留成自己的性格。

## 特性

- 精简系统提示词
- 将所有 Border 改为 2px
- 构建配置设为最优，同时编译时长最长
- 更改品牌为 Zed Vela
- 树图上下文管理
- 独立的 LLM 提供商配置面板
- 支持多个相同提供商
- 将编辑器的工具栏变得紧凑

## 行为

与官方 Zed 混用时，请留意以下数据归属：

- 用户目录未改名（Windows 下为 `%APPDATA%\Zed` 与 `%LOCALAPPDATA%\Zed`），settings.json、键位、线程数据库、扩展与语言服务器均与官方共用
- 单实例互斥锁与更新通道相互独立，两版可以同时运行，但会并发写同一批文件，以最后写入者为准
- 树状上下文仅 Vela 读写；官方版本保存同一线程时会丢弃分支结构，只保留当前分支
- 兼容提供商的 API Key 按提供商名称独立保存，与官方按 URL 保存的密钥互不覆盖；旧密钥首次加载时自动迁移，在 Vela 中重置则会连带清理旧条目
- 如需完全隔离，可用 `--user-data-dir` 指定独立的数据目录

## 开发

- [在 macOS 上构建](./docs/src/development/macos.md)
- [在 Linux 上构建](./docs/src/development/linux.md)
- [在 Windows 上构建](./docs/src/development/windows.md)

## 许可

Zed Vela 源自 Zed，源代码主要遵循 GPL-3.0-or-later，部分标注为 Apache-2.0 的组件以其标注为准。感谢 Zed Industries 与开源社区的每一份馈赠。
