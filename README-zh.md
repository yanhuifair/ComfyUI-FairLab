# ComfyUI-FairLab

<p align="center">
  <img src="https://img.shields.io/badge/ComfyUI-custom__nodes-blue?style=flat-square&logo=python" alt="ComfyUI Custom Nodes">
  <img src="https://img.shields.io/badge/version-1.0.90-green?style=flat-square" alt="Version">
  <img src="https://img.shields.io/badge/license-MIT-yellow?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/nodes-58-orange?style=flat-square" alt="Nodes Count">
  <a href="https://github.com/yanhuifair/ComfyUI-FairLab/stargazers"><img src="https://img.shields.io/github/stars/yanhuifair/ComfyUI-FairLab?style=flat-square" alt="GitHub stars"></a>
</p>

<p align="center">
  58 个 ComfyUI 实用节点集合，涵盖字符串处理、图像 I/O 与操作、<br>
  算术逻辑、PBR 材质工具链及工作流诊断。<br>
  全部节点遵循 ComfyUI <code>IO.*</code> 类型标注规范。
</p>

> 语言： [English](./README.md)

---

## 目录

- [ComfyUI-FairLab](#comfyui-fairlab)
  - [目录](#目录)
  - [概述](#概述)
  - [安装](#安装)
    - [前置条件](#前置条件)
    - [方式一：ComfyUI Manager（推荐）](#方式一comfyui-manager推荐)
    - [方式二：手动安装](#方式二手动安装)
    - [验证安装](#验证安装)
    - [更新](#更新)
    - [依赖项](#依赖项)
    - [常见问题](#常见问题)
  - [节点参考](#节点参考)
    - [String（字符串，16 个）](#string字符串16-个)
    - [Image（图像，26 个）](#image图像26-个)
      - [I/O 与加载](#io-与加载)
      - [图像处理](#图像处理)
      - [视频](#视频)
      - [调制](#调制)
      - [PBR 贴图](#pbr-贴图)
    - [Logic（逻辑，11 个）](#logic逻辑11-个)
    - [Utility（工具，5 个）](#utility工具5-个)
  - [技术说明](#技术说明)
  - [License](#license)

---

## 概述

ComfyUI-FairLab 补充了 ComfyUI 核心节点集中缺失但生产工作流中常用的操作。

| 类别   | 数量 | 范围 |
|:-------|:----:|:-----|
| String | 16   | 字符串构造、标签操作、翻译、编码修复、文件读写 |
| Image  | 26   | 文件/URL/Base64 加载、批量读写、视频转换、Alpha 与通道操作、PBR 贴图、调制 |
| Logic  | 11   | 算术运算符、条件分支、类型转换 |
| Utility| 5    | 诊断打印、宽高比预设、双 LoRA 加载、沙箱脚本 |

---

## 安装

### 前置条件

- 已安装并正常运行 **ComfyUI**。如未安装，请参考[官方指南](https://github.com/comfyanonymous/ComfyUI#installing)。
- **Python** 3.10+（与 ComfyUI 使用同一环境）。
- **Git**（手动安装时需要）。

### 方式一：ComfyUI Manager（推荐）

1. 打开 ComfyUI，进入 **Manager** 面板。
2. 点击 **Install Custom Nodes**。
3. 搜索 `ComfyUI-FairLab`。
4. 点击 **Install**，安装完成后重启 ComfyUI。

### 方式二：手动安装

```bash
# 1. 进入 custom_nodes 目录
cd ComfyUI/custom_nodes

# 2. 克隆仓库
git clone https://github.com/yanhuifair/ComfyUI-FairLab.git

# 3. 进入项目目录
cd ComfyUI-FairLab

# 4. 安装依赖
pip install -r requirements.txt

# 5. 重启 ComfyUI
```

### 验证安装

重启 ComfyUI 后，双击画布空白处，搜索 `FairLab` 或任意节点名（如 `String Append`、`Load Image Batch From Directory`）。如果搜索结果中出现对应节点，说明安装成功。

### 更新

**ComfyUI Manager：** 在 Manager 面板点击 **Update All**。

**手动更新：**
```bash
cd ComfyUI/custom_nodes/ComfyUI-FairLab
git pull
pip install -r requirements.txt
```

### 依赖项

| 包名 | 用途 |
|:-----|:-----|
| `googletrans`                      | String Translate 翻译节点 |
| `opencv-python`                    | 视频与图像互转；通用图像处理 |
| `requests`                         | Download Image、Load Image From URL |
| `nest_asyncio`                     | 异步事件循环支持 |
| `perfect-pixel[opencv]>=0.1.4`     | Perfect Pixel 节点 |

### 常见问题

**安装后节点未显示**

- 确认安装后已重启 ComfyUI。
- 检查 ComfyUI 终端中是否有与 `ComfyUI-FairLab` 相关的导入错误。
- 验证依赖是否已安装：`pip list | grep -E "opencv|googletrans|perfect-pixel"`

**ImportError: No module named 'cv2'**

```bash
pip install opencv-python
```

**Google Translate 翻译不可用**

`googletrans` 库可能需要更新版本，尝试：
```bash
pip install --upgrade googletrans
```

---

## 节点参考

### String（字符串，16 个）

字符串构造、持久化及面向标签的操作。

| 节点 | 说明 |
|:-----|:-----|
| **String**                 | 输出常量字符串 |
| **Int**                    | 输出常量整数值 |
| **Float**                  | 输出常量浮点值 |
| **String Append**          | 将多个输入字符串拼接为单一输出 |
| **Load String**            | 从本地文本文件读取字符串内容 |
| **Save String To Directory**   | 将字符串写入磁盘文件 |
| **Load String From Directory** | 从指定目录中的文件读取字符串 |
| **Show String**            | 在节点 UI 面板上显示字符串值，便于检查 |
| **String Translate**       | 通过 Google Translate 翻译字符串（源语言/目标语言可配置） |
| **Fix UTF-8 String**       | 修复因编码不匹配导致的 UTF-8 乱码（如畸变字符） |
| **Range String**           | 在数值范围内生成带索引的字符串序列（如 `frame_001`..`frame_100`） |
| **Prepend Tags**           | 在逗号分隔的标签列表中为每个标签添加前缀 |
| **Append Tags**            | 在逗号分隔的标签列表中为每个标签添加后缀 |
| **Exclude Tags**           | 从标签列表中移除指定标签 |
| **Unique Tags**            | 去除标签列表中的重复项 |
| **ASCII Art Text**         | 使用系统字体将文本渲染为 ASCII 艺术图像 |

### Image（图像，26 个）

#### I/O 与加载

| 节点 | 说明 |
|:-----|:-----|
| **Load Image From Directory**       | 从目录加载单张图像 |
| **Load Image Batch From Directory** | 将目录中所有图像加载为批次张量 |
| **Load Image From URL**             | 直接从远程 URL 获取图像并转为张量 |
| **Download Image**                  | 下载远程图像并持久化到本地磁盘 |
| **Save Image To Directory**         | 将图像张量写入指定输出路径 |
| **Save Image To Folder**            | 按子目录组织保存图像 |
| **Image To Base64**                 | 将图像张量编码为 Base64 字符串 |
| **Base64 To Image**                 | 将 Base64 字符串解码为图像张量 |

#### 图像处理

| 节点 | 说明 |
|:-----|:-----|
| **Resize Image**           | 将图像张量缩放至目标尺寸 |
| **Image Size**             | 将图像宽高输出为标量值 |
| **Image Shape**            | 输出图像批次的完整张量形状 `[B, C, H, W]` |
| **Image Remove Alpha**     | 从 RGBA 图像中移除 Alpha 通道，输出 RGB |
| **Fill Alpha**             | 为 RGB 图像添加或替换 Alpha 通道 |
| **Pure Color Image**       | 生成指定尺寸和 RGB 值的纯色图像 |
| **Images Range**           | 按起止索引从批次中截取子图像集 |
| **Images Index**           | 按索引从批次中提取单张图像 |
| **Images Cat**             | 沿批次维度拼接多个图像批次 |
| **Outpainting Pad**        | 为 outpainting 工作流填充图像边框 |
| **Perfect Pixel**          | 无亚像素插值的整数倍缩放（精确倍数的最近邻采样） |

#### 视频

| 节点 | 说明 |
|:-----|:-----|
| **Video To Image**   | 从视频文件中提取帧作为图像批次张量 |
| **Image To Video**   | 将图像批次合成为视频文件（可配置帧率） |

#### 调制

| 节点 | 说明 |
|:-----|:-----|
| **Modulation**            | 应用误差扩散调制（Floyd–Steinberg 风格半色调处理） |
| **Modulation Direction**  | 定向误差扩散，可配置扫描方向和速度 |

#### PBR 贴图

| 节点 | 说明 |
|:-----|:-----|
| **Mask Map**                  | 将灰度图像转换为标准化遮罩张量 |
| **Detail Map**                | 从输入图像生成细节贴图，用于 PBR 材质管线 |
| **Roughness To Smoothness**   | 粗糙度与光滑度贴图互转（取反） |

### Logic（逻辑，11 个）

用于节点图层级计算的算术与控制流原语。

| 节点 | 说明 |
|:-----|:-----|
| **Number**         | 通用数值常量（INT / FLOAT 可切换） |
| **Add**            | `A + B`，自动适配 INT/FLOAT 类型 |
| **Subtract**       | `A - B` |
| **Multiply**       | `A × B`，自动适配类型 |
| **Multiply Int**   | `A × B`，结果强制转为 INT |
| **Divide**         | `A ÷ B` |
| **Max**            | `max(A, B)` |
| **Min**            | `min(A, B)` |
| **If**             | 条件分支：根据布尔条件将输出路由至 A 或 B |
| **Float To Int**   | FLOAT 转 INT（截断） |
| **Int To Float**   | INT 转 FLOAT |

### Utility（工具，5 个）

诊断、预设与扩展。

| 节点 | 说明 |
|:-----|:-----|
| **Print Any**        | 将任意输入值打印到控制台，用于调试 |
| **Print Image**      | 将图像张量元数据打印到控制台（shape、dtype、值范围） |
| **Aspect Ratios**    | 常用宽高比下拉选择器（1:1、16:9、4:3、3:2、2:3、21:9） |
| **Load LoRA Dual**   | 同时加载两个 LoRA 模型，各自独立调节强度 |
| **Python Script**    | 在 AST 白名单沙箱中求值受限 Python 表达式；支持算术、比较运算符及一组精选内置函数（`abs`、`min`、`max`、`round`、`len` 等）。`exec` 和 `eval` 被显式禁用。 |

---

## 技术说明

**Python Script 沙箱**

表达式求值器使用 AST 白名单，仅允许显式注册的运算符（`+`、`-`、`*`、`/`、`//`、`%`、`**`、`<`、`>`、`<=`、`>=`、`==`、`!=`、`is`、`is not`、`not`、一元 `+`/`-`）及内置函数（`abs`、`bool`、`float`、`int`、`len`、`max`、`min`、`round`、`sorted`、`str`、`sum`、`tuple`、`list`）。无法通过 `exec` 或 `eval` 执行任意代码。

**IO 类型标注**

所有节点均使用 ComfyUI `IO.*` 类型标注（`IO.STRING`、`IO.IMAGE`、`IO.INT`、`IO.FLOAT` 等），与现代 ComfyUI 类型系统兼容。

**ASCII Art Text**

依赖系统字体支持。在最小化 Linux 环境中需安装字体包（如 Debian/Ubuntu 下的 `fonts-dejavu-core`）。

**视频操作**

帧提取与合成依赖 `opencv-python`。处理时间与分辨率和帧数成正比。

---

## License

MIT — 详见 [LICENSE](./LICENSE)。

Copyright (c) 2025 [Fair](https://github.com/yanhuifair)
