@echo off
cd d:\jindai
ssh share "bash -c 'source ~/.zshrc; cd ~/jindai-ui; npm run build && cp -r dist ~/jindai/ui'"
scp -r share:~/jindai/ui . > nul
