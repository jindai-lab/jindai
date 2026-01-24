"""Config file"""
import os
import sys
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConfigObject(BaseModel):
    # 允许额外字段：这是扩展性的核心
    # 未在类中定义的字段会被保存在对象实例中，通过 model_extra 访问
    model_config = ConfigDict(extra='allow')

    # 已定义字段
    secret_key: str
    concurrent: int = 3
    storage: str
    database: str
    redis: str
    plugins: List[str] = Field(default_factory=lambda: ["*"])
    oidc_secrets: str
    port: int = 8370
    embedding_model: str
    embedding_dims: int
    
    ui_dist: str
    paddle_remote: str
    
    constants: dict = Field(default_factory=dict)
    
    @field_validator('redis', mode='after')
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip('/')
    
    @classmethod
    def load_from_yaml(cls, file_path: str) -> "ConfigObject":
        """加载并解析 YAML"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        print('* loaded config from', file_path)
        return cls.model_validate(data)

    def get_extra(self) -> Dict[str, Any]:
        """获取所有未在类中定义的配置项"""
        return self.model_extra or {}


if '-c' in sys.argv:
    config_arg = sys.argv.index('-c')
    config_file = sys.argv[config_arg + 1]
    sys.argv.pop(config_arg + 1)
    sys.argv.pop(config_arg)
    os.environ['CONFIG_FILE'] = config_file


instance = ConfigObject.load_from_yaml(os.environ.get('CONFIG_FILE', 'config.yaml'))