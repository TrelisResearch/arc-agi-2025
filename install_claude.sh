# 1) (Usually already there) Install curl
apt-get update -y && apt-get install -y curl

# 2) Install Claude Code native binary
curl -fsSL https://claude.ai/install.sh | bash

# # 3) Make sure it's on PATH for this shell and future shells
# echo 'export PATH="$HOME/.claude/bin:$PATH"' >> ~/.bashrc
# export PATH="$HOME/.claude/bin:$PATH"

# 4) Verify
claude --version