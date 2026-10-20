import unittest
from unittest.mock import patch

import httpx
from langgraph_sdk.client import LangGraphClient
from orchestrator.demo import DemoCoordinator, MockWorkers
from decomposer import create_decomposer_agent

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from orchestrator.compaction import CompactionError, ContextCompaction, SUMMARY_PROMPT


class RecordingModel(BaseChatModel):
    requests: list = Field(default_factory=list)
    fail_summary: bool = False

    @property
    def _llm_type(self):
        return 'compaction-test'

    def _generate(self, messages, **kwargs):
        raise NotImplementedError

    async def _agenerate(self, messages, **kwargs):
        self.requests.append(messages)
        summarizing = messages[0].text == SUMMARY_PROMPT
        if summarizing and self.fail_summary:
            raise RuntimeError('offline')
        text = 'Goal: fix /project/app.py. Agent worker-1 is running; wait for run-1.' if summarizing else 'answer'
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


class SummarizingDemo(DemoCoordinator):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if messages[0].text == SUMMARY_PROMPT:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content='Earlier project inspection. No active agents.'))])
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class CompactionTests(unittest.IsolatedAsyncioTestCase):
    async def test_original_decomposer_tool_cycle_after_compaction(self):
        model = SummarizingDemo()
        workers = MockWorkers()
        async with httpx.AsyncClient(transport=httpx.MockTransport(workers.handle), base_url='http://mock.local') as http:
            with patch('decomposer.core.get_client', return_value=LangGraphClient(http)):
                agent = create_decomposer_agent(
                    decomposer_model=model,
                    agent_types=[{'agent_type_id': 'mock', 'description': 'mock', 'assistant_id': 'mock', 'url': 'http://mock.local'}],
                    middleware=[ContextCompaction(model, trigger_tokens=1000, keep_tokens=400, summary_tokens=200)],
                    agent_run_budget_seconds=None,
                )
                result = await agent.ainvoke({'messages': [HumanMessage(content='old history ' * 500), AIMessage(content='previous answer'), HumanMessage(content='hello')], 'agents': {}, 'agent_runs': {}})
                self.assertEqual(result['messages'][-1].text, 'hello')
                self.assertEqual(result['context_compactions'], 1)
                self.assertEqual(len(result['agents']), 1)
                self.assertEqual([m.name for m in result['messages'] if isinstance(m, ToolMessage)], ['new', 'run', 'wait'])
                self.assertEqual(result['messages'][0].text, 'old history ' * 500)

    async def test_graph_retains_archive_and_checkpoints_compact_view(self):
        model = RecordingModel()
        middleware = ContextCompaction(model, trigger_tokens=500, keep_tokens=180, summary_tokens=100)
        memory = InMemorySaver()
        agent = create_agent(model, system_prompt='Original evolved instructions.', middleware=[middleware], checkpointer=memory)
        config = {'configurable': {'thread_id': 'workspace'}}
        first = HumanMessage(content='old details ' * 250)
        await agent.ainvoke({'messages': [first]}, config)
        await agent.ainvoke({'messages': [HumanMessage(content='What next?')]}, config)
        state = (await agent.aget_state(config)).values
        self.assertEqual(len(state['messages']), 4)
        self.assertEqual(state['messages'][0].text, first.text)
        self.assertEqual(state['context_compactions'], 1)
        request = model.requests[-1]
        self.assertIsInstance(request[0], SystemMessage)
        self.assertEqual(request[0].text, 'Original evolved instructions.')
        self.assertIn('worker-1', request[1].text)
        self.assertNotIn(first.text, [m.text for m in request])
        self.assertEqual(request[-1].text, 'What next?')
        # Rebuild middleware, as happens after changing the selected model.
        agent = create_agent(model, middleware=[ContextCompaction(model, trigger_tokens=500, keep_tokens=180, summary_tokens=100)], checkpointer=memory)
        await agent.ainvoke({'messages': [HumanMessage(content='Continue.')]}, config)
        self.assertIn('worker-1', model.requests[-1][0].text)
        self.assertEqual((await agent.aget_state(config)).values['context_compactions'], 1)

    async def test_tool_exchanges_never_split(self):
        middleware = ContextCompaction(RecordingModel(), trigger_tokens=100, keep_tokens=50, summary_tokens=20)
        messages = [HumanMessage(content='old ' * 200),
                    AIMessage(content='', tool_calls=[{'id': 'a', 'name': 'new', 'args': {}}, {'id': 'b', 'name': 'run', 'args': {}}]),
                    ToolMessage(content='created', tool_call_id='a'),
                    ToolMessage(content='running', tool_call_id='b'),
                    AIMessage(content='done'), HumanMessage(content='next')]
        self.assertIn(middleware.cutoff(messages, 0), (1, 4, 5))
        self.assertEqual(middleware.cutoff(messages[:-2], 0), 1)
        # A large pending exchange cannot be compacted in pieces.
        messages[1].content = 'large ' * 200
        self.assertEqual(middleware.cutoff(messages[:3], 0), 0)
        self.assertEqual(middleware.cutoff(messages[:4], 0), 4)

    async def test_failure_keeps_checkpoint_and_allows_retry(self):
        model = RecordingModel()
        middleware = ContextCompaction(model, trigger_tokens=500, keep_tokens=180, summary_tokens=100)
        agent = create_agent(model, middleware=[middleware], checkpointer=InMemorySaver())
        config = {'configurable': {'thread_id': 'failure'}}
        await agent.ainvoke({'messages': [HumanMessage(content='details ' * 400)]}, config)
        model.fail_summary = True
        with self.assertRaises(CompactionError):
            await agent.ainvoke({'messages': [HumanMessage(content='next')]}, config)
        state = (await agent.aget_state(config)).values
        self.assertEqual(len(state['messages']), 3)
        self.assertNotIn('context_summary', state)
        model.fail_summary = False
        await agent.ainvoke(None, config)
        self.assertEqual((await agent.aget_state(config)).values['context_compactions'], 1)

    async def test_large_history_chunking_reuses_previous_summary(self):
        model = RecordingModel()
        middleware = ContextCompaction(model, trigger_tokens=500, keep_tokens=180, summary_tokens=100, chunk_chars=500)
        agent = create_agent(model, middleware=[middleware])
        result = await agent.ainvoke({'messages': [HumanMessage(content='details ' * 400), AIMessage(content='ok'), HumanMessage(content='next')]})
        summaries = [m for m in model.requests if m[0].text == SUMMARY_PROMPT]
        self.assertGreater(len(summaries), 2)
        self.assertIn('worker-1', summaries[1][1].text)
        self.assertEqual(len(result['messages']), 4)


if __name__ == '__main__':
    unittest.main()
