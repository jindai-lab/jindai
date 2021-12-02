from queue import Queue
import traceback
import threading

    
class Plugin:
    
    def __init__(self, app, **config) -> None:
        self.config = config
        self.app = app
        
    def get_tools(self):
        return []
    
    def get_special_pages(self):
        return []
    
    def get_callbacks(self):
        return []
    
    def run_tool(self, ctx, name, *args, **kwargs):
        name = name.replace('-', '_')
        return getattr(self, name)(ctx, *args, **kwargs)
    
    def run_callback(self, ctx, name, *args, **kwargs):
        name = name.replace('-', '_') + '_callback'
        return getattr(self, name)(ctx, *args, **kwargs)
        
    def special_pages(self, rs, params, orders_params, **vars):
        return [], {}, {}
    
    
class PluginContext:
    
    def __init__(self, name='', bundle={}, logging_hook=None, join=False) -> None:
        self.name = name
        self.bundle = bundle
        self.logging_hook = logging_hook
        self.queue = Queue()
        self.alive = True
        self.join = join
        self.ret = None
        
    def log(self, *args):
        s = self.name + '|' + ' '.join(map(str, args))
        print(s)
        self.queue.put(s)
        
    def run(self, func, *args, **kwargs):
        def _run():
            try:
                self.ret = func(self, *args, **kwargs)
            except Exception as ex:
                self.log('Error:', ex)
                self.log(traceback.format_exc())
            self.alive = False
        
        thr = threading.Thread(target=_run)
        thr.start()
        if self.join: thr.join()
    
    def fetch(self):
        while not self.queue.empty():
            yield self.queue.get() + '\n'
