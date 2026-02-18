#!/bin/bash
# deploy.sh — 云端一键部署脚本
# 在云服务器上执行：curl -sSL <你的脚本地址> | bash
# 或者 git clone 后直接运行 ./deploy.sh

set -e

echo "=========================================="
echo "  小来训练 — 云端部署"
echo "=========================================="

if ! command -v docker &> /dev/null; then
    echo "[1/4] 安装 Docker..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable --now docker
else
    echo "[1/4] Docker 已安装，跳过"
fi

if ! command -v docker compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "[2/4] 安装 Docker Compose..."
    apt-get update && apt-get install -y docker-compose-plugin
else
    echo "[2/4] Docker Compose 已安装，跳过"
fi

echo "[3/4] 构建镜像..."
docker compose build

echo "[4/4] 启动循环训练（后台运行）..."
docker compose up -d

echo ""
echo "=========================================="
echo "  部署完成！小来正在训练中"
echo "=========================================="
echo ""
echo "常用命令："
echo "  查看训练日志：docker compose logs -f"
echo "  停止训练：    docker compose down"
echo "  重启训练：    docker compose restart"
echo "  查看状态：    docker compose ps"
echo ""
echo "模型文件保存在 ./checkpoints/ 目录"
