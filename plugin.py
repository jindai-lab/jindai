class Plugin:
    
    def __init__(self, app, **config) -> None:
        self.config = config
        self.app = app
        
    def get_special_pages(self):
        return []
    
    def special_pages(self, ds, post_args):
        return [], {}, {}

    def get_callbacks(self):
        return []
        
    def run_callback(self, name, *args, **kwargs):
        name = name.replace('-', '_') + '_callback'
        return getattr(self, name)(*args, **kwargs)
        