# 构建 nanoGPT

这个仓库包含了从零开始复现 [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master) 的完整过程。Git 提交历史特意保持了逐步清晰的结构，这样任何人都可以轻松浏览提交历史，看到项目是如何慢慢构建起来的。此外，在 [YouTube 上有配套的视频讲座](https://youtu.be/l8pRSuU81PU)，您可以看到我介绍每个提交并逐步解释各个部分。

我们基本上从一个空文件开始，一步步构建到完整复现 [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (124M) 模型。如果您有更多耐心或资金，这些代码也可以复现 [GPT-3](https://arxiv.org/pdf/2005.14165) 模型。虽然 GPT-2 (124M) 模型在当年（2019年，约5年前）可能需要训练相当长时间，但如今复现它只需要约1小时和约10美元。如果您没有足够强大的GPU，需要云GPU服务器，我推荐 [Lambda](https://lambdalabs.com)。

需要注意的是，GPT-2 和 GPT-3 都是简单的语言模型，在互联网文档上训练，它们所做的就是"梦想"互联网文档。因此，这个仓库/视频不涵盖聊天微调，您无法像与 ChatGPT 对话那样与它对话。微调过程（虽然概念上相当简单 - SFT 只是交换数据集并继续训练）在这部分之后，将在稍后的时间涵盖。目前，如果您在100亿个标记的训练后用"Hello, I'm a language model,"提示124M模型，它会说这样的话：

```
Hello, I'm a language model, and my goal is to make English as easy and fun as possible for everyone, and to find out the different grammar rules
Hello, I'm a language model, so the next time I go, I'll just say, I like this stuff.
Hello, I'm a language model, and the question is, what should I do if I want to be a teacher?
Hello, I'm a language model, and I'm an English person. In languages, "speak" is really speaking. Because for most people, there's
```

在400亿个标记的训练后：

```
Hello, I'm a language model, a model of computer science, and it's a way (in mathematics) to program computer programs to do things like write
Hello, I'm a language model, not a human. This means that I believe in my language model, as I have no experience with it yet.
Hello, I'm a language model, but I'm talking about data. You've got to create an array of data: you've got to create that.
Hello, I'm a language model, and all of this is about modeling and learning Python. I'm very good in syntax, however I struggle with Python due
```

哈哈。无论如何，一旦视频发布，这里也将是常见问题解答的地方，以及修复和勘误表的地方，我相信会有很多 :)

关于讨论和问题，请使用 [讨论板块](https://github.com/karpathy/build-nanogpt/discussions)，如需更快的交流，请查看我的 [Zero To Hero Discord](https://discord.gg/3zy8kqD9Cp)，**#nanoGPT** 频道：

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## 视频

[让我们复现 GPT-2 (124M) YouTube 讲座](https://youtu.be/l8pRSuU81PU)

## 勘误表

小的清理工作，我们忘记在切换到 flash attention 后删除 bias 的 `register_buffer`，通过最近的 PR 修复了。

早期版本的 PyTorch 可能在从 uint16 转换为 long 时遇到困难。在 `load_tokens` 内部，我们添加了 `npt = npt.astype(np.int32)` 来使用 numpy 将 uint16 转换为 int32，然后再转换为 torch tensor 并转换为 long。

`torch.autocast` 函数接受一个参数 `device_type`，我试图固执地传递 `device` 希望能正常工作，但 PyTorch 实际上真的只想要类型并在某些 PyTorch 版本中创建错误。所以我们希望例如设备 `cuda:3` 被简化为 `cuda`。目前，设备 `mps`（Apple Silicon）会变成 `device_type` CPU，我不是100%确定这是 PyTorch 的预期方式。

令人困惑的是，`model.require_backward_grad_sync` 实际上被前向和后向传播都使用。将这行代码向上移动，以便它也应用于前向传播。

## 生产环境

对于与 nanoGPT 非常相似的更多生产级运行，我推荐查看以下仓库：

- [litGPT](https://github.com/Lightning-AI/litgpt)
- [TinyLlama](https://github.com/jzhang38/TinyLlama)

## 常见问题

## 许可证

MIT
