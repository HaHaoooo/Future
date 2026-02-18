# 小来云端训练运维手册

## 目录

- [架构概览](#架构概览)
- [前置准备](#前置准备)
- [首次部署](#首次部署)
- [日常运维](#日常运维)
- [tmux 操作速查](#tmux-操作速查)
- [模型同步（云端 ↔ 本地）](#模型同步云端--本地)
- [更新代码到云端](#更新代码到云端)
- [故障排查](#故障排查)

---

## 架构概览

```
本地 Mac                          Google Cloud VM (future)
┌──────────┐    scp / git push    ┌──────────────────────┐
│  代码开发  │ ──────────────────→ │  Docker 容器          │
│  模型调试  │ ←────────────────── │  └─ 循环训练 (loop)   │
└──────────┘    scp 拉回模型       │  tmux 持久终端        │
                                  └──────────────────────┘
```

- **云端**：Docker 容器运行循环训练，`restart: unless-stopped` 保证服务器重启后自动恢复
- **tmux**：保持终端会话，SSH 断开后仍可重连查看
- **checkpoints/**：挂载卷，容器重建数据不丢

---

## 前置准备

| 项目 | 值 |
|------|----|
| 服务器 | Google Cloud VM `future`，区域 `us-central1-a` |
| 外部 IP | `34.42.211.131`（以实际为准，可能会变） |
| 用户 | `admin` |
| 项目目录 | `~/Future` |

---

## 首次部署

### 1. SSH 登录云服务器

通过 Google Cloud Console 网页点「SSH」按钮，或本地安装 gcloud 后：

```bash
gcloud compute ssh future --zone=us-central1-a
```

### 2. 上传代码

**方式 A — GitHub（仓库需为 Public）：**

```bash
sudo apt update && sudo apt install -y git
git clone https://github.com/HaHaoooo/Future.git
cd Future
```

**方式 B — 本地直接上传（通过 Google Cloud Console 网页 SSH 的上传功能）：**

本地先打包：

```bash
cd ~
tar czf Future.tar.gz -C ~/Future \
    --exclude='.venv' --exclude='__pycache__' \
    --exclude='.git' --exclude='.idea' --exclude='.DS_Store' .
```

通过浏览器 SSH 窗口右上角齿轮 → Upload file 上传 `Future.tar.gz`，然后在服务器上：

```bash
mkdir -p ~/Future && tar xzf ~/Future.tar.gz -C ~/Future
cd ~/Future
```

### 3. 一键部署

```bash
sudo ./deploy.sh
```

脚本会自动完成：安装 Docker → 安装 Docker Compose → 构建镜像 → 启动循环训练。

### 4. 安装 tmux 并查看训练

```bash
sudo apt install -y tmux
tmux new -s xiaolai
sudo docker compose logs -f
```

---

## 日常运维

### 查看训练状态

```bash
# SSH 登录后
tmux attach -t xiaolai              # 重连到 tmux 会话
# 或者直接看日志
sudo docker compose logs -f         # 实时日志（Ctrl+C 退出查看，不影响训练）
sudo docker compose logs --tail 50  # 最近 50 行
```

### 停止训练

```bash
sudo docker compose down
```

### 重启训练

```bash
sudo docker compose restart
```

### 查看容器状态

```bash
sudo docker compose ps
```

### 重新构建并启动（代码更新后）

```bash
sudo docker compose up -d --build
```

---

## tmux 操作速查

tmux 让你的终端会话在 SSH 断开后继续存在。

### 核心操作

| 操作 | 命令 / 按键 |
|------|-------------|
| 新建会话 | `tmux new -s xiaolai` |
| 重连会话 | `tmux attach -t xiaolai` |
| 断开会话（不关闭） | `Ctrl+B` 然后按 `D` |
| 列出所有会话 | `tmux ls` |
| 关闭会话 | `tmux kill-session -t xiaolai` |

### 窗口内操作

| 操作 | 按键 |
|------|------|
| 进入滚动模式 | `Ctrl+B` 然后按 `[` |
| 滚动 | 方向键 ↑↓ 或 PageUp / PageDown |
| 退出滚动模式 | `Q` |
| 搜索 | 滚动模式下按 `Ctrl+S`（向下）或 `Ctrl+R`（向上） |

### 典型工作流

```
SSH 登录
  → tmux attach -t xiaolai      # 重连，看到实时训练画面
  → 查看完毕
  → Ctrl+B, D                   # 断开 tmux（训练继续）
  → exit                        # 退出 SSH（训练继续）
```

---

## 模型同步（云端 ↔ 本地）

### 从云端拉回训练好的模型到本地

**方式 A — gcloud（推荐）：**

```bash
gcloud compute scp --zone=us-central1-a \
    admin@future:~/Future/checkpoints/xiaolai.npz \
    ~/Future/checkpoints/xiaolai.npz

gcloud compute scp --zone=us-central1-a \
    admin@future:~/Future/checkpoints/xiaolai_meta.json \
    ~/Future/checkpoints/xiaolai_meta.json
```

**方式 B — 浏览器下载：**

在服务器上：

```bash
# 打包模型
tar czf ~/checkpoints.tar.gz -C ~/Future/checkpoints .
```

通过 Google Cloud Console 网页 SSH 窗口右上角齿轮 → Download file → 输入 `/home/admin/checkpoints.tar.gz`。

本地解压：

```bash
tar xzf ~/Downloads/checkpoints.tar.gz -C ~/Future/checkpoints/
```

### 把本地模型上传到云端

```bash
gcloud compute scp --zone=us-central1-a \
    ~/Future/checkpoints/xiaolai.npz \
    admin@future:~/Future/checkpoints/xiaolai.npz

gcloud compute scp --zone=us-central1-a \
    ~/Future/checkpoints/xiaolai_meta.json \
    admin@future:~/Future/checkpoints/xiaolai_meta.json

# 上传后重启训练让模型生效
ssh admin@future "cd ~/Future && sudo docker compose restart"
```

---

## 更新代码到云端

本地改完代码后，把更新同步到云端：

**方式 A — GitHub：**

```bash
# 本地
cd ~/Future
git add . && git commit -m "update" && git push

# 云端
cd ~/Future
git pull
sudo docker compose up -d --build
```

**方式 B — 直接上传：**

本地打包：

```bash
tar czf Future.tar.gz -C ~/Future \
    --exclude='.venv' --exclude='__pycache__' \
    --exclude='.git' --exclude='.idea' --exclude='.DS_Store' \
    --exclude='checkpoints' .
```

上传后在云端：

```bash
tar xzf ~/Future.tar.gz -C ~/Future
cd ~/Future
sudo docker compose up -d --build
```

---

## 故障排查

### 容器没在运行

```bash
sudo docker compose ps                # 查看状态
sudo docker compose logs --tail 100   # 看最近日志找报错
sudo docker compose up -d             # 重启
```

### 磁盘满了

```bash
df -h                                  # 查看磁盘使用
sudo docker system prune -f           # 清理无用镜像和缓存
```

### 想完全重新构建

```bash
sudo docker compose down
sudo docker compose build --no-cache
sudo docker compose up -d
```

### 查看容器内部

```bash
sudo docker exec -it xiaolai-train bash    # 进入容器
ls checkpoints/                            # 检查模型文件
exit                                       # 退出容器
```

### 服务器重启后

不需要操作。`docker-compose.yml` 配置了 `restart: unless-stopped`，Docker 会自动重启训练容器。

验证：

```bash
sudo docker compose ps    # 确认状态是 running
```
