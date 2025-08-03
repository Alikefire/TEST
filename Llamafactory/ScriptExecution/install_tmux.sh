#!/bin/bash

#===================================================
# 一键安装 bison + libevent + ncurses + tmux
# 安装路径: $HOME/local
#===================================================

set -e  # 出错立即退出

# 版本定义
BISON_VERSION="3.8.2"
LIBEVENT_VERSION="2.1.12-stable"
NCURSES_VERSION="6.4"
TMUX_VERSION="3.4"

# 安装路径
PREFIX="$HOME/local"

# 创建必要目录
mkdir -p $PREFIX/{bin,lib,include,src}
cd $PREFIX/src

# 配置环境变量
export PATH="$PREFIX/bin:$PATH"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig"
export CPPFLAGS="-I$PREFIX/include"
export LDFLAGS="-L$PREFIX/lib"

echo "==== 安装目录: $PREFIX ===="

#===================================================
# 1. 安装 bison
#===================================================
if ! command -v bison >/dev/null 2>&1; then
    echo ">>> 下载并编译 bison $BISON_VERSION ..."
    wget -q http://ftp.gnu.org/gnu/bison/bison-$BISON_VERSION.tar.gz
    tar -xvzf bison-$BISON_VERSION.tar.gz
    cd bison-$BISON_VERSION
    ./configure --prefix=$PREFIX
    make -j$(nproc)
    make install
    cd ..
else
    echo ">>> 检测到系统已存在 bison，跳过安装"
fi

#===================================================
# 2. 安装 libevent
#===================================================
echo ">>> 下载并编译 libevent $LIBEVENT_VERSION ..."
wget -q https://github.com/libevent/libevent/releases/download/release-$LIBEVENT_VERSION/libevent-$LIBEVENT_VERSION.tar.gz
tar -xvzf libevent-$LIBEVENT_VERSION.tar.gz
cd libevent-$LIBEVENT_VERSION
./configure --prefix=$PREFIX --disable-shared
make -j$(nproc)
make install
cd ..

#===================================================
# 3. 安装 ncurses
#===================================================
echo ">>> 下载并编译 ncurses $NCURSES_VERSION ..."
wget -q https://ftp.gnu.org/pub/gnu/ncurses/ncurses-$NCURSES_VERSION.tar.gz
tar -xvzf ncurses-$NCURSES_VERSION.tar.gz
cd ncurses-$NCURSES_VERSION
./configure --prefix=$PREFIX
make -j$(nproc)
make install
cd ..

#===================================================
# 4. 安装 tmux
#===================================================
echo ">>> 下载并编译 tmux $TMUX_VERSION ..."
wget -q https://github.com/tmux/tmux/releases/download/$TMUX_VERSION/tmux-$TMUX_VERSION.tar.gz
tar -xvzf tmux-$TMUX_VERSION.tar.gz
cd tmux-$TMUX_VERSION

# 强制使用 bison 作为 yacc
YACC="bison -y" ./configure --prefix=$PREFIX
make -j$(nproc)
make install
cd ..

#===================================================
# 配置 PATH
#===================================================
if ! grep -q "export PATH=\$HOME/local/bin:\$PATH" $HOME/.bashrc; then
    echo 'export PATH=$HOME/local/bin:$PATH' >> $HOME/.bashrc
fi

echo ">>> 安装完成，请运行以下命令加载 tmux:"
echo "    source ~/.bashrc"
echo "    tmux -V"

