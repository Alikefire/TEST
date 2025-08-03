#!/bin/bash

#===================================================
# 一键卸载 bison + libevent + ncurses + tmux
# 删除 $HOME/local 下的所有相关文件
#===================================================

PREFIX="$HOME/local"

echo ">>> 开始卸载 tmux 及其依赖..."

# 删除安装目录
if [ -d "$PREFIX" ]; then
    echo ">>> 删除 $PREFIX 目录..."
    rm -rf "$PREFIX"
else
    echo ">>> 未找到 $PREFIX 目录，无需删除"
fi

# 删除 PATH 配置行
if grep -q "export PATH=\$HOME/local/bin:\$PATH" $HOME/.bashrc; then
    echo ">>> 移除 .bashrc 中的 PATH 配置"
    sed -i '/export PATH=\$HOME\/local\/bin:\$PATH/d' $HOME/.bashrc
fi

# 删除历史源码文件
SRC_DIR="$HOME/local/src"
if [ -d "$SRC_DIR" ]; then
    echo ">>> 删除源码目录 $SRC_DIR"
    rm -rf "$SRC_DIR"
fi

echo ">>> 卸载完成！请重新加载 bashrc："
echo "    source ~/.bashrc"
