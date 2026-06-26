# ComfyUI-FairLab

<p align="center">
  <img src="https://img.shields.io/badge/ComfyUI-custom__nodes-blue?style=flat-square&logo=python" alt="ComfyUI Custom Nodes">
  <img src="https://img.shields.io/badge/version-1.0.89-green?style=flat-square" alt="Version">
  <img src="https://img.shields.io/badge/license-MIT-yellow?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/nodes-58-orange?style=flat-square" alt="Nodes Count">
  <a href="https://github.com/yanhuifair/ComfyUI-FairLab/stargazers"><img src="https://img.shields.io/github/stars/yanhuifair/ComfyUI-FairLab?style=flat-square" alt="GitHub stars"></a>
</p>

<p align="center"><strong>58 个实用节点集合</strong>，涵盖字符串处理、图像操作、逻辑运算、工具调试等领域，全部采用 ComfyUI 新版 <code>IO.*</code> 节点类型规范。</p>

---

## 📦 安装

### 方法一：通过 ComfyUI Manager（推荐）

在 ComfyUI Manager 中搜索 `ComfyUI-FairLab`，点击安装即可。

### 方法二：手动安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yanhuifair/ComfyUI-FairLab.git
cd ComfyUI-FairLab
pip install -r requirements.txt
```

### 依赖项

| 包名 | 用途 |
|------|------|
| `googletrans` | 字符串翻译节点 |
| `opencv-python` | 图像/视频处理 |
| `requests` | 网络下载节点 |
| `nest_asyncio` | 异步事件循环支持 |
| `perfect-pixel[opencv]>=0.1.4` | Perfect Pixel 像素级图像处理 |

---

## 🧩 节点列表（共 58 个）

### 🔤 字符串 (String) — 16 个

| 节点名 | 说明 |
|--------|------|
| **String** | 输出固定字符串 |
| **Int** | 输出整数值 |
| **Float** | 输出浮点值 |
| **String Append** | 拼接多个字符串 |
| **Load String** | 从文件加载字符串内容 |
| **Save String To Directory** | 保存字符串到文件 |
| **Load String From Directory** | 从目录加载字符串文件 |
| **Show String** | 将字符串显示到 UI 面板 |
| **String Translate** | 使用 Google 翻译字符串 |
| **Fix UTF-8 String** | 修复 UTF-8 编码异常字符 |
| **Range String** | 按范围生成序号字符串 |
| **Prepend Tags** | 在标签前追加前缀 |
| **Append Tags** | 在标签后追加后缀 |
| **Exclude Tags** | 从标签列表中排除指定标签 |
| **Unique Tags** | 去除重复标签 |
| **ASCII Art Text** | 将文本渲染为 ASCII 艺术图像 |

### 🖼️ 图像 (Image) — 26 个

| 节点名 | 说明 |
|--------|------|
| **Load Image From Directory** | 从目录加载单张图像 |
| **Load Image Batch From Directory** | 从目录批量加载图像 |
| **Load Image From URL** | 从 URL 加载图像 |
| **Download Image** | 下载网络图像到本地 |
| **Save Image To Directory** | 保存图像到指定目录 |
| **Save Image To Folder** | 保存图像到文件夹（带子目录管理） |
| **Resize Image** | 调整图像尺寸 |
| **Image Size** | 获取图像尺寸信息 |
| **Image Shape** | 获取图像形状张量 |
| **Image To Base64** | 图像转 Base64 编码 |
| **Base64 To Image** | Base64 解码为图像 |
| **Image Remove Alpha** | 移除图像 Alpha 通道 |
| **Fill Alpha** | 填充 Alpha 通道 |
| **Pure Color Image** | 生成纯色图像 |
| **Images Range** | 按范围取子图像集 |
| **Images Index** | 按索引取单张图像 |
| **Images Cat** | 沿批次维度拼接图像 |
| **Video To Image** | 视频文件转图像序列 |
| **Image To Video** | 图像序列转视频文件 |
| **Modulation** | 图像调制/混合处理 |
| **Modulation Direction** | 定向调制处理 |
| **Outpainting Pad** | 外扩画布补边 |
| **Mask Map** | 遮罩映射转换 |
| **Detail Map** | 细节贴图生成 |
| **Roughness To Smoothness** | 粗糙度转光滑度贴图 |
| **Perfect Pixel** | 像素级精确缩放处理 |

### 🧮 逻辑 (Logic) — 11 个

| 节点名 | 说明 |
|--------|------|
| **Number** | 通用数值节点 |
| **Add** | 加法运算（支持 INT/FLOAT） |
| **Subtract** | 减法运算 |
| **Multiply** | 乘法运算（支持 INT/FLOAT） |
| **Multiply Int** | 整数乘法 |
| **Divide** | 除法运算 |
| **Max** | 取最大值 |
| **Min** | 取最小值 |
| **If** | 条件判断分支 |
| **Float To Int** | 浮点转整数 |
| **Int To Float** | 整数转浮点 |

### 🛠️ 工具 (Utility) — 5 个

| 节点名 | 说明 |
|--------|------|
| **Print Any** | 打印任意类型数据到控制台 |
| **Print Image** | 打印图像信息到控制台 |
| **Aspect Ratios** | 常用宽高比预设列表 |
| **Load LoRA Dual** | 双 LoRA 加载器 |
| **Python Script** | 安全的 Python 表达式脚本节点 |

---

## ⚠️ 注意事项

- **Python Script** 节点使用受限表达式求值器，不支持任意 `exec`/`eval`，保障安全性。
- 所有节点均遵循 ComfyUI 新版 `IO.*` 数据类型规范。
- ASCII Art Text 节点依赖系统字体渲染，Linux 环境可能需要安装基础字体包。

---

## 📄 License

MIT © [Fair](https://github.com/yanhuifair)

---

<p align="center"><sub>Made with ❤️ for the ComfyUI community</sub></p>
