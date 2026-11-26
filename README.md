# langchain-bilinc

LangChain integration for [Bilinc](https://bilinc.space), shared and versioned memory for AI agents.

- `BilincRemember`: a tool that saves a fact, decision or preference to Bilinc (`commit`).
- `BilincRecall`: a tool that searches Bilinc memory in plain language (`recall`).
- `BilincRetriever`: a retriever that returns recalled memories as LangChain `Document`s.

All three wrap the official [`bilinc`](https://pypi.org/project/bilinc/) Python SDK.

Source: <https://github.com/ReARCLabs/bilinc-langchain>

## Install

```bash
pip install -U langchain-bilinc
```

## Credentials

Set `BILINC_API_KEY`, or pass `api_key=`. If neither is set, the SDK uses the key saved by `bilinc login`. Get a key at [bilinc.space/signup](https://bilinc.space/signup).

```bash
export BILINC_API_KEY="..."
```

## Tools

```python
from langchain_bilinc import BilincRecall, BilincRemember

remember = BilincRemember(key_prefix="support-agent")
remember.invoke({"key": "user.timezone", "value": "Europe/Istanbul"})
# {"saved": True, "key": "support-agent.user.timezone", "entry_version": "..."}

recall = BilincRecall(limit=5)
recall.invoke({"query": "what timezone is the user in?"})
# [{"key": "support-agent.user.timezone", "value": "Europe/Istanbul", "score": ..., ...}]
```

Give both to an agent so it can save and look up memory across sessions:

```python
from langchain.agents import create_agent
from langchain_bilinc import BilincRecall, BilincRemember

agent = create_agent(
    model="anthropic:claude-sonnet-5-5",
    tools=[BilincRecall(), BilincRemember(key_prefix="my-agent")],
    system_prompt="Recall memory before acting on earlier decisions. Save only durable facts.",
)
```

`BilincRemember` details:

- Writing to an existing key replaces its value; Bilinc keeps the earlier value in that memory's history.
- Without a `key`, the tool derives a stable key from the value, so saving the same value twice writes the same key.
- Options: `key_prefix`, `default_memory_type` (`semantic`), `default_importance` (`1.0`), `source` (`"langchain"`), `session_id`, `metadata`.

`BilincRecall` options: `limit` (`5`), `profile` (`fast`, `balanced`, `verified`, `deep`; higher profiles depend on your plan), `memory_types`.

## Retriever

```python
from langchain_bilinc import BilincRetriever

retriever = BilincRetriever(k=5, profile="balanced")
docs = retriever.invoke("what did we decide about the release?")
for doc in docs:
    print(doc.metadata["key"], doc.page_content)
```

`page_content` is the stored value; JSON values are serialized. `metadata` holds `key`, `memory_type`, `importance`, `score`, `source`, `entry_version` and `updated_at` when the service returns them.

Bilinc's limits are applied before the call:
- `k` (on the retriever or per call) is capped at 100.
- The query is cut at 4,096 characters.
- A blank query returns no documents.

A failed recall is logged as a warning and raised as `BilincRetrieverError`, with the SDK error as its cause.

## Errors and costs

Tool errors are raised as `ToolException` (set `handle_tool_error=True` to return them to the model). A failed save means nothing was saved. Saves and recalls use Bilinc credits; see [pricing](https://bilinc.space/pricing).

## Development

```bash
pip install -e . pytest pytest-asyncio "langchain-tests>=1.1.9,<2"
pytest
```

The unit tests replace the SDK transport with a fake and never reach the network.

## License

MIT for this package. The `bilinc` SDK it depends on has its own license (BUSL-1.1).
