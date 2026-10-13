"""Keep only the latest result from each independent ROI worker."""
from dataclasses import replace
from .live import LiveRecognitionUpdate


class MultiLiveWorker:
    def __init__(self, support, target):
        self.support, self.target = support, target
        self.detector, self.session = support.detector, support.session
        self.latest = [None,None]

    def start(self):
        self.support.start()
        try:
            self.target.start()
        except Exception:
            self.support.stop()
            raise

    def stop(self, timeout=3):
        if timeout<0:raise ValueError('timeout cannot be negative')
        # Signal both before waiting for either worker to finish.
        for worker in (self.support,self.target):
            if hasattr(worker,'request_stop'):worker.request_stop()
        try:
            self.support.stop(timeout)
        finally:
            self.target.stop(timeout)

    def request_template_update(self, specs):
        self.support.request_template_update(specs)

    def take_latest(self):
        changed=False
        for index,worker in enumerate((self.support,self.target)):
            update=worker.take_latest()
            if update is not None:
                self.latest[index]=update
                changed=True
        if not changed:
            return None
        primary=self.latest[0] or LiveRecognitionUpdate(0,0,None,(),'캐릭터 캡처 대기')
        target=self.latest[1]
        return replace(primary, target_update=target)
