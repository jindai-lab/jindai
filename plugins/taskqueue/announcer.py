"""Message announcer"""

from queue import Queue, Full
from threading import Thread
import time


class MessageAnnouncer:
    """Message announcer"""

    def __init__(self):
        self.listeners = []

    def listen(self):
        """listen to the announcer"""
        message_queue = Queue(maxsize=5)
        self.listeners.append(message_queue)
        self.pulse()
        return message_queue
    
    def _heartbeat(self):
        while self.listeners:
            self.announce('pulse')
            time.sleep(10)
    
    def pulse(self):
        Thread(target=self._heartbeat).start()

    def announce(self, msg):
        """Announce message"""
        for i in reversed(range(len(self.listeners))):
            try:
                self.listeners[i].put_nowait(msg)
            except Full:
                del self.listeners[i]

    def log(self, *args):
        """Announce log"""
        self.announce(' '.join([str(_) for _ in args]))

    def get_log(self, prefix):
        """Get prefixed log"""
        return lambda *args: self.log(prefix, '|', *args)


announcer = MessageAnnouncer()
