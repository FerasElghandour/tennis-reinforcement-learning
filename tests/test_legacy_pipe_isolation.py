"""Exercise the actual vendored servicer class without optional Unity imports."""
import ast
from multiprocessing import Pipe
from pathlib import Path

def test_live_pipe_endpoints_are_owned_by_each_environment():
    source=Path(__file__).resolve().parents[1]/'vendor/unityagents/rpc_communicator.py'
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='UnityToExternalServicerImplementation')
    module=ast.Module(body=[node],type_ignores=[]);namespace={'Pipe':Pipe,'UnityToExternalServicer':object}
    exec(compile(module,str(source),'exec'),namespace)
    cls=namespace[node.name];first=cls();second=cls()
    try:
        assert first.parent_conn is not second.parent_conn
        first.parent_conn.send('first');assert first.child_conn.recv()=='first'
        assert not second.child_conn.poll(.01)
        first.parent_conn.close();first.child_conn.close()
        second.parent_conn.send('still open');assert second.child_conn.recv()=='still open'
    finally:
        for obj in [first,second]:
            obj.parent_conn.close();obj.child_conn.close()
