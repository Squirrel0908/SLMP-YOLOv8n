#!/bin/bash

# 1. 定义源文件和备份目录
SOURCE_FILE="/hy-tmp/runs/prune/exp2-0.001/slc-yolov8n-groupsl-0.001-prune/weights/last.pt"
BACKUP_DIR="/hy-tmp/runs/prune/exp2-0.001/slc-yolov8n-groupsl-0.001-prune/weights/backups"

# 2. 创建备份文件夹
mkdir -p $BACKUP_DIR

echo "🚀 自动备份哨兵已启动..."

# 3. 循环备份
# 按照你目前的 3.12 it/s 速度，100 轮大约需要 20-30 分钟
# 我们设置每 30 分钟备份一次，这样肯定能抓到不同阶段的权重
while true
do
    if [ -f "$SOURCE_FILE" ]; then
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        cp "$SOURCE_FILE" "$BACKUP_DIR/sl_checkpoint_$TIMESTAMP.pt"
        echo "✅ 已备份当前权重: sl_checkpoint_$TIMESTAMP.pt"
    else
        echo "⏳ 等待 last.pt 生成..."
    fi
    
    # 每 1800 秒（30分钟）执行一次
    sleep 1800
done