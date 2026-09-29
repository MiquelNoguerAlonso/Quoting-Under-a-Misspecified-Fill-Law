"""Normalized research engine. No live gateway, market data, or protocol decoding.

Times, tick prices and quantities are integers. Reports are authoritative
cumulative-fill snapshots with monotonically increasing per-order sequence.
A terminal snapshot includes ALL cumulative fills. Adapters must establish
that contract before using this ledger with an external protocol.
"""
from dataclasses import dataclass
import heapq


@dataclass
class Child:
    size: int
    side: int = 1
    cumulative: int = 0
    reserved: int = 0
    sequence: int = -1
    status: str = 'sent'


class Ledger:
    def __init__(self, parent=None, inventory=0, limit=100):
        self.parent = parent
        self.inventory = inventory
        self.limit = limit
        self.children = {}
        self.filled = 0
        self.recovery = False

    def outstanding(self, side=None):
        return sum(c.reserved for c in self.children.values()
                   if side is None or c.side == side)

    def invariant(self):
        assert all(0 <= c.cumulative <= c.size and c.reserved >= 0
                   for c in self.children.values())
        if self.parent is not None:
            assert 0 <= self.filled <= self.parent
            assert self.filled + self.outstanding() <= self.parent
        assert self.inventory + self.outstanding(1) <= self.limit
        assert self.inventory - self.outstanding(-1) >= -self.limit

    def submit(self, key, size, side=1):
        if self.recovery:
            raise ValueError('unresolved report gap')
        if key in self.children or type(size) is not int or size <= 0 or side not in (-1,1):
            raise ValueError('invalid child')
        if self.parent is not None and (side != 1 or self.filled+self.outstanding()+size > self.parent):
            raise ValueError('parent overcommitment')
        if side == 1 and self.inventory+self.outstanding(1)+size > self.limit:
            raise ValueError('buy exposure')
        if side == -1 and self.inventory-self.outstanding(-1)-size < -self.limit:
            raise ValueError('sell exposure')
        self.children[key] = Child(size=size, side=side, reserved=size)
        self.invariant()

    def cancel_request(self, key):
        c = self.children[key]
        if c.status != 'terminal':
            c.status = 'cancel_requested'
        self.invariant()  # NO release on request.

    def report(self, key, sequence, cumulative, terminal=False):
        c = self.children[key]
        if sequence <= c.sequence:
            return False  # duplicate or older normalized snapshot
        if c.status == 'terminal':
            raise ValueError('new report after terminal requires reconciliation')
        if not c.cumulative <= cumulative <= c.size:
            raise ValueError('invalid cumulative execution')
        delta = cumulative-c.cumulative
        c.cumulative = cumulative
        c.sequence = sequence
        self.filled += delta if self.parent is not None else 0
        self.inventory += c.side*delta
        c.reserved = 0 if terminal else c.size-cumulative
        if terminal:
            c.status = 'terminal'
        elif c.status != 'cancel_requested':
            c.status = 'live'
        self.invariant()
        return True


class Queue:
    """One displayed FIFO level. External cancels target known order IDs."""
    def __init__(self):
        self.orders = []

    def add(self, key, size):
        if type(size) is not int or size <= 0 or any(k == key for k,_ in self.orders):
            raise ValueError('invalid queue add')
        self.orders.append([key,size])

    def ahead(self, key):
        quantity=0
        for k,v in self.orders:
            if k == key:
                return quantity
            quantity += v
        raise KeyError(key)

    def cancel(self, key, size=None):
        for i,(k,v) in enumerate(self.orders):
            if k == key:
                n = v if size is None else size
                if not 0 <= n <= v:
                    raise ValueError('cancel size')
                if n == v:
                    self.orders.pop(i)
                else:
                    self.orders[i][1] -= n
                return n
        return 0

    def consume(self, size):
        if type(size) is not int or size < 0:
            raise ValueError('consume size')
        fills=[]
        while size and self.orders:
            key,v = self.orders[0]
            f=min(v,size)
            fills.append((key,f))
            size-=f
            if f == v:
                self.orders.pop(0)
            else:
                self.orders[0][1]-=f
        return fills


class EventLoop:
    """Deterministic scheduler. Equal-time events use explicit priority then insertion order."""
    def __init__(self):
        self.now=0
        self.events=[]
        self.counter=0
        self.trace=[]
        self.available={}

    def schedule(self, time, kind, callback, priority=0):
        if type(time) is not int or time < self.now:
            raise ValueError('invalid event time')
        self.counter+=1
        heapq.heappush(self.events,(time,priority,self.counter,kind,callback))

    def observe(self, exchange_time, receipt_time, key, value):
        if receipt_time < exchange_time:
            raise ValueError('receipt before exchange event')
        def receive():
            self.available[key]=value
        self.schedule(receipt_time,'receipt:'+key,receive)

    def run(self):
        while self.events:
            time,_,_,kind,callback=heapq.heappop(self.events)
            self.now=time
            callback()
            self.trace.append({'time':time,'event':kind,'available':dict(self.available)})


def cancel_race():
    """Submit at t=0, cancel at t=2, partial fill at t=3, effective cancel t=5.
    Fill report reaches local ledger at 4, terminal snapshot at 7.
    """
    engine=EventLoop(); ledger=Ledger(parent=10); queue=Queue()
    ledger.submit('ours',6)
    cumulative=[0]
    reservations=[]
    def add():
        queue.add('ahead',2); queue.add('ours',6)
        engine.schedule(2,'accept_report',lambda:ledger.report('ours',0,0),priority=-1)
    def request():
        ledger.cancel_request('ours')
        reservations.append(ledger.outstanding())
    def trade():
        for key,qty in queue.consume(5):
            if key == 'ours': cumulative[0]+=qty
        amount=cumulative[0]
        engine.schedule(4,'fill_report',lambda:ledger.report('ours',1,amount))
    def effective_cancel():
        queue.cancel('ours')
        amount=cumulative[0]
        engine.schedule(7,'cancel_terminal',lambda:ledger.report('ours',2,amount,True))
    engine.schedule(1,'exchange_accept',add)
    engine.schedule(2,'send_cancel',request)
    engine.schedule(3,'exchange_trade',trade)
    engine.schedule(5,'exchange_cancel',effective_cancel)
    engine.observe(1,6,'book',101)
    def decision():
        assert 'book' not in engine.available
        assert ledger.outstanding() == 3
    engine.schedule(5,'local_decision',decision,priority=1)
    engine.run()
    assert ledger.filled == 3 and ledger.outstanding() == 0
    assert reservations == [6]
    return {'filled':ledger.filled,'reservation_on_request':reservations[0],
            'final_reserved':ledger.outstanding(),'trace':engine.trace}
