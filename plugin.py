class Plugin:
    
    def __init__(self, app, **config) -> None:
        self.config = config
        self.app = app
        
    def get_special_pages(self):
        return []
    
    def special_pages(self, ds, post_args):
        return [], {}, {}
