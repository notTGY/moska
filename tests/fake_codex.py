#!/usr/bin/env python3
"""Deterministic stdio peer for bridge tests; never runs model/tool code."""
import json
import os
import sys
import threading
import time
from uuid import uuid4

if '--version' in sys.argv:
    print('codex-cli 0.163.0')
    raise SystemExit

lock = threading.Lock()
threads = {}
turns = {}
queued = set()


def send(message):
    with lock:
        print(json.dumps(message), flush=True)


def event(method, **params):
    send({'method': method, 'params': params})


for line in sys.stdin:
    request = json.loads(line)
    if 'method' not in request:
        if request.get('id') == 'approval-test':
            assert request['result'] == {'decision': 'decline'}
            event('test/denied')
        elif request.get('id') == 'unknown-test':
            assert request['error']['code'] == -32601
            event('test/unknownDenied')
        continue
    method, params = request['method'], request.get('params', {})
    result = {}
    if method == 'initialize':
        assert params['clientInfo']['name'] == 'moska'
        result = {'userAgent': 'fake'}
    elif method == 'thread/start':
        assert params['approvalPolicy'] == 'never'
        assert params['sandbox'] == 'workspace-write'
        assert params['cwd'] == os.getcwd()
        thread = str(uuid4())
        threads[thread] = []
        result = {'thread': {'id': thread}}
    elif method == 'thread/fork':
        thread = str(uuid4())
        threads[thread] = list(threads[params['threadId']])
        result = {'thread': {'id': thread}}
    elif method == 'turn/start':
        assert params['sandboxPolicy']['type'] == 'workspaceWrite'
        assert params['sandboxPolicy']['networkAccess'] is False
        assert params['sandboxPolicy']['writableRoots'] == [os.getcwd()]
        assert params['approvalPolicy'] == 'never'
        assert params['cwd'] == os.getcwd()
        thread = params['threadId']
        turn = str(uuid4())
        turns[thread] = turn
        prompt = params['input'][0]['text']
        if prompt == 'queued':
            queued.add(thread)
        previous = list(threads[thread])
        threads[thread].append(prompt)
        event('turn/started', threadId=thread, turn={'id': turn, 'status': 'inProgress'})
        if prompt == 'crash':
            os._exit(4)
        if prompt == 'slow-start':
            time.sleep(.2)
        item = {'id': 'cmd-' + turn, 'type': 'commandExecution', 'command': 'pwd',
                'cwd': os.getcwd(), 'status': 'inProgress'}
        event('item/started', threadId=thread, turnId=turn, item=item)
        item = {**item, 'status': 'completed', 'aggregatedOutput': os.getcwd(), 'exitCode': 0}
        event('item/completed', threadId=thread, turnId=turn, item=item)
        # Deliver completion BEFORE the turn/start ack to exercise routing races.
        if prompt not in ('hold', 'slow-start', 'queued'):
            if prompt == 'fail':
                final = {'id': turn, 'status': 'failed', 'error': {'message': 'worker failure'}, 'items': []}
            else:
                text = json.dumps({'cwd': os.getcwd(), 'previous': previous, 'prompt': prompt})
                answer = {'id': 'answer-' + turn, 'type': 'agentMessage', 'phase': 'final_answer', 'text': text}
                event('item/completed', threadId=thread, turnId=turn, item=answer)
                final = {'id': turn, 'status': 'completed', 'items': []}
            event('turn/completed', threadId=thread, turn=final)
        result = {'turn': {'id': turn, 'status': 'inProgress', 'items': []}}
    elif method == 'turn/interrupt':
        assert turns[params['threadId']] == params['turnId']
        if params['threadId'] in queued:
            queued.remove(params['threadId'])
            send({'id': request['id'], 'error': {'code': -32600, 'message': 'no active turn to interrupt'}})
            continue
        event('turn/completed', threadId=params['threadId'], turn={'id': params['turnId'], 'status': 'interrupted', 'items': []})
    elif method == 'test/approval':
        send({'id': 'approval-test', 'method': 'item/commandExecution/requestApproval', 'params': {'command': 'bad'}})
        send({'id': 'unknown-test', 'method': 'future/privilegedRequest', 'params': {}})
    elif method == 'test/no-response':
        continue
    if 'id' in request:
        send({'id': request['id'], 'result': result})
