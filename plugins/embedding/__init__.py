"""内置向量嵌入插件（plugins/embedding）。

按功能把原先散在核心里的嵌入相关组件收拢为一个内置插件：

* ``worker.EmbeddingWorkerPlugin`` —— 常驻补算 worker：轮询
  ``embedding_pending_queue``，有待补段落就调用
  ``update_text_embeddings()`` 分批算向量落库；带磁盘水位
  守卫（pause/resume 滞回）与空闲指数退避；同时提供 ``/embeddings`` 路由。
* ``local_backend`` —— ``embedding_api_base: local`` 时的本机 ONNX 后端
  （原 ``jindai/local_embedding.py``），被 ``jindai.embedutils.EmbeddingClient``
  在运行时按需加载。

worker 的启停由 app lifespan → ``PluginManager.startup()/shutdown()`` 统一管理。
"""

from .worker import EmbeddingWorkerPlugin

__all__ = ["EmbeddingWorkerPlugin"]
