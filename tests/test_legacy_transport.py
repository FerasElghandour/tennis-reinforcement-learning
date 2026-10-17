import ast
from pathlib import Path
from types import SimpleNamespace
import pytest

def communicator_class():
    path=Path(__file__).resolve().parents[1]/'vendor/unityagents/rpc_communicator.py'
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='RpcCommunicator')
    class Message:
        def __init__(self):self.header=SimpleNamespace(status=None);self.unity_input=SimpleNamespace(CopyFrom=lambda _:None)
    ns={'Communicator':object,'UnityInput':object,'UnityOutput':object,'UnityMessage':Message,'UnityTimeOutException':TimeoutError}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    return ns['RpcCommunicator']

def test_exchange_times_out_before_an_unbounded_receive():
    client=communicator_class()()
    class Pipe:
        def send(self,_):pass
        def poll(self,timeout):assert timeout==30;return False
        def recv(self):raise AssertionError('Must not block after an elapsed timeout')
    client.unity_to_external=SimpleNamespace(parent_conn=Pipe())
    with pytest.raises(TimeoutError,match='step response'):client.exchange(object())

def test_close_stops_server_and_closes_both_owned_endpoints_idempotently():
    client=communicator_class()();events=[]
    class Pipe:
        def close(self):events.append('closed')
    class Server:
        def stop(self,grace):events.append(('stop',grace));return SimpleNamespace(wait=lambda timeout:events.append(('wait',timeout)))
    client.server=Server();client.unity_to_external=SimpleNamespace(parent_conn=Pipe(),child_conn=Pipe())
    client.close();client.close()
    assert events.count(('stop',0))==1 and ('wait',5) in events
    assert events.count('closed')==4 and not client.is_open
