# Jindai

A platform for digital humanities research, made by and for humanities researchers.

For code relating to the frontend, please refer to [jindai-ui](https://github.com/jindai-lab/jindai-ui).

For plugins to make full use of the platform, please follow [this link](https://pan.baidu.com/s/11_QT5mG1gw7mNjz23EQGGA?pwd=s8gq).

Refer to [wiki](https://github.com/jindai-lab/jindai/wiki) for further information.

License: MIT


## Installation

```bash
git clone https://github.com/jindai-lab/jindai/
cd jindai
cp config.yaml.sample config.yaml
cp docker-compose.yaml.sample docker-compose.yaml
# IMPORTANT! EDIT config.yaml AND docker-compose.yaml FIRST
nano config.yaml docker-compose.yaml
wget https://github.com/jindai-lab/jindai-ui/releases/download/v0.2.0/dist.tgz
mkdir ui && tar xzf dist.tgz -C ui && rm dist.tgz
```

### Install with Docker
```bash
docker build . -t jindai
docker compose up -d
```

### Install on local machine
```bash
pip install -r requirements.txt
python3 -m jindai web-service
```

### Run init script
```bash
docker exec -it -w /app jindai python3 -m jindai init
```

☕ [Buy me a coffee](https://www.buymeacoffee.com/zhuth90)
