import json
import unittest
from unittest.mock import patch

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph_sdk.client import LangGraphClient

from adapters.providers.openai import ConnectionError, OpenAIConnection
from adapters.providers.openai_model import ChatGPTModel
from decomposer import create_decomposer_agent
from orchestrator.demo import MockWorkers

REAL_CLIENT = httpx.AsyncClient


class FakeConnection(OpenAIConnection):
    async def access_token(self):
        return 'fake-token'


class ChatGPTModelTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_decomposer_cycle_over_stateless_responses(self):
        workers, requests = MockWorkers(), []
        responses = [
            {'type': 'function_call', 'namespace': 'moska', 'name': 'new',
             'call_id': 'new-call', 'arguments': json.dumps({'agent_type_id': 'mock'})},
            None,
            {'type': 'function_call', 'namespace': 'moska', 'name': 'wait',
             'call_id': 'wait-call', 'arguments': '{}'},
            {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'hello'}]},
        ]

        def handle(request):
            if request.url.host == 'mock.local':
                return workers.handle(request)
            payload = json.loads(request.content)
            requests.append(payload)
            index = len(requests) - 1
            item = responses[index]
            if index == 1:
                output = next(i for i in payload['input'] if i.get('type') == 'function_call_output')
                agent_id = json.loads(output['output'])['agent_id']
                item = {'type': 'function_call', 'namespace': 'moska', 'name': 'run', 'call_id': 'run-call',
                        'arguments': json.dumps({'agent_id': agent_id, 'prompt': 'test task'})}
            response = {'model': 'test-model', 'output': [
                {'type': 'reasoning', 'id': f'reason-{index}', 'encrypted_content': 'opaque-reasoning', 'summary': []}, item,
            ]}
            events = [
                {'type': 'response.output_item.done', 'output_index': i, 'item': item}
                for i, item in enumerate(response['output'])
            ]
            response['output'] = []
            events.append({'type': 'response.completed', 'response': response})
            content = ''.join('data: ' + json.dumps(event) + '\n\n' for event in events)
            return httpx.Response(200, content=content, headers={'content-type': 'text/event-stream'})

        transport = httpx.MockTransport(handle)
        with patch('adapters.providers.openai_model.httpx.AsyncClient', side_effect=lambda **kw: REAL_CLIENT(transport=transport, **kw)):
            async with REAL_CLIENT(transport=transport, base_url='http://mock.local') as http:
                with patch('decomposer.core.get_client', return_value=LangGraphClient(http)):
                    model = ChatGPTModel(connection=FakeConnection(), model_name='test-model', reasoning_effort='high')
                    agent = create_decomposer_agent(decomposer_model=model, agent_types=[{
                        'agent_type_id': 'mock', 'description': 'mock', 'assistant_id': 'mock', 'url': 'http://mock.local',
                    }], agent_run_budget_seconds=None)
                    result = await agent.ainvoke({'messages': [HumanMessage(content='test task')], 'agents': {}, 'agent_runs': {}})
        self.assertEqual(result['messages'][-1].content, 'hello')
        self.assertEqual(len(requests), 4)
        self.assertTrue(all(p['reasoning'] == {'effort': 'high'} for p in requests))
        self.assertTrue(all(p['stream'] and not p['store'] for p in requests))
        self.assertTrue(all(p['tools'][0]['type'] == 'namespace' for p in requests))
        self.assertTrue(any(i.get('encrypted_content') == 'opaque-reasoning' for i in requests[-1]['input']))
        output = next(i for i in requests[-1]['input'] if i.get('call_id') == 'wait-call' and i['type'] == 'function_call_output')
        self.assertEqual(json.loads(output['output'])[0]['response'], 'hello')
        self.assertNotIn('previous_response_id', requests[-1])
        self.assertFalse(any(i.get('role') == 'system' for i in requests[0]['input']))

    async def test_system_prompt_text_is_preserved(self):
        prompt = '  Original instructions.\n\nKeep exact whitespace.  '
        instructions, items = ChatGPTModel._input([SystemMessage(content=prompt), HumanMessage(content='hi')])
        self.assertEqual(instructions, prompt)
        self.assertEqual(items, [{'role': 'user', 'content': 'hi'}])

    async def test_incomplete_failed_or_truncated_stream_is_not_success(self):
        for event in ({'type': 'response.incomplete'}, {'type': 'response.failed'}, {'type': 'error'}, None):
            with self.subTest(event=event):
                content = 'data: ' + json.dumps(event) + '\n\n' if event else 'data: [DONE]\n\n'
                transport = httpx.MockTransport(lambda request: httpx.Response(200, content=content))
                with patch('adapters.providers.openai_model.httpx.AsyncClient', side_effect=lambda **kw: REAL_CLIENT(transport=transport, **kw)):
                    model = ChatGPTModel(connection=FakeConnection(), model_name='test-model')
                    with self.assertRaises(ConnectionError):
                        await model.ainvoke('hi')


if __name__ == '__main__':
    unittest.main()
