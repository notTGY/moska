"""Async LangChain adapter for ChatGPT-plan Responses requests."""

import json

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from .openai import ConnectionError, OpenAIConnection, RESOURCE


class ChatGPTModel(BaseChatModel):
    connection: OpenAIConnection = Field(exclude=True, repr=False)
    model_name: str
    reasoning_effort: str | None = None

    @property
    def _llm_type(self):
        return 'chatgpt-plan-responses'

    @property
    def _identifying_params(self):
        return {'model_name': self.model_name, 'reasoning_effort': self.reasoning_effort}

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        functions = [
            {'type': 'function', **convert_to_openai_tool(tool, strict=True)['function']}
            for tool in tools
        ]
        if isinstance(tool_choice, str) and tool_choice not in ('auto', 'none', 'required'):
            functions = [tool for tool in functions if tool['name'] == tool_choice]
            if not functions:
                raise ConnectionError('The requested orchestrator tool is unavailable.')
            tool_choice = 'required'
        namespaces = [{
            'type': 'namespace', 'name': 'moska',
            'description': 'Agent orchestration tools.', 'tools': functions,
        }] if functions else []
        return self.bind(tools=namespaces, tool_choice=tool_choice, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError('ChatGPTModel requires async invocation.')

    @staticmethod
    def _input(messages):
        instructions, items = [], []
        for message in messages:
            if isinstance(message, SystemMessage):
                instructions.append(message.text)
            elif isinstance(message, HumanMessage):
                items.append({'role': 'user', 'content': message.text})
            elif isinstance(message, ToolMessage):
                items.append({
                    'type': 'function_call_output', 'call_id': message.tool_call_id,
                    'output': message.content if isinstance(message.content, str) else json.dumps(message.content),
                })
            elif isinstance(message, AIMessage):
                raw = message.additional_kwargs.get('response_items')
                if raw is not None:
                    # Preserve encrypted reasoning alongside calls in stateless requests.
                    items.extend(raw)
                else:
                    if message.text:
                        items.append({'role': 'assistant', 'content': message.text})
                    for call in message.tool_calls:
                        items.append({
                            'type': 'function_call', 'namespace': 'moska',
                            'name': call['name'], 'call_id': call['id'],
                            'arguments': json.dumps(call['args']),
                        })
        return '\n\n'.join(instructions), items

    @staticmethod
    def _result(response):
        text, calls = [], []
        output = response.get('output', [])
        for item in output:
            if item['type'] == 'message':
                for part in item.get('content', []):
                    if part['type'] == 'output_text':
                        text.append(part['text'])
                    elif part['type'] == 'refusal':
                        text.append(part['refusal'])
            elif item['type'] == 'function_call':
                if item.get('namespace') not in (None, 'moska'):
                    raise ConnectionError('OpenAI returned a tool from an unexpected namespace.')
                arguments = json.loads(item['arguments'])
                if not isinstance(arguments, dict):
                    raise ConnectionError('OpenAI returned invalid tool arguments.')
                calls.append({'name': item['name'], 'args': arguments, 'id': item['call_id']})
        message = AIMessage(
            content='\n'.join(text), tool_calls=calls,
            additional_kwargs={'response_items': output},
            response_metadata={'model_name': response.get('model')},
        )
        if not text and not calls:
            raise ConnectionError('OpenAI returned an empty response. Try again.')
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        token = await self.connection.access_token()
        instructions, items = self._input(messages)
        payload = {
            'model': self.model_name, 'instructions': instructions, 'input': items,
            'store': False, 'stream': True, 'include': ['reasoning.encrypted_content'],
        }
        if self.reasoning_effort is not None:
            payload['reasoning'] = {'effort': self.reasoning_effort}
        if kwargs.get('tools'):
            payload['tools'] = kwargs['tools']
        if kwargs.get('tool_choice') is not None:
            payload['tool_choice'] = kwargs['tool_choice']
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=20)) as client:
                async with client.stream(
                    'POST', RESOURCE + '/responses', json=payload,
                    headers={'Authorization': 'Bearer ' + token},
                ) as stream:
                    if stream.is_error:
                        raise ConnectionError(f'OpenAI rejected the model request (HTTP {stream.status_code}). Try /model or /connect again.')
                    output_items = {}
                    async for line in stream.aiter_lines():
                        if not line.startswith('data:'):
                            continue
                        data = line[5:].strip()
                        if data == '[DONE]':
                            break
                        event = json.loads(data)
                        if event.get('type') == 'response.output_item.done':
                            output_items[event['output_index']] = event['item']
                        if event.get('type') == 'response.completed':
                            response = event['response']
                            # Streaming completion envelopes may omit the items
                            # already delivered through output_item.done events.
                            if not response.get('output'):
                                response['output'] = [output_items[index] for index in sorted(output_items)]
                            return self._result(response)
                        if event.get('type') in ('response.failed', 'response.incomplete', 'error'):
                            raise ConnectionError('OpenAI could not complete the response. Check your plan usage or try again.')
            raise ConnectionError('OpenAI disconnected before completing the response. Try again.')
        except httpx.HTTPError:
            raise ConnectionError('Could not complete the OpenAI request. Check your connection and try again.') from None
        except (KeyError, TypeError, ValueError):
            raise ConnectionError('OpenAI returned an invalid response.') from None
