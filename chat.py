import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent


def print_messages(response):
    messages = response["messages"]

    for message in messages:

        if isinstance(message, HumanMessage):
            emoji = "👤"
        elif isinstance(message, AIMessage):
            emoji = "🤖"
        elif isinstance(message, ToolMessage):
            emoji = "🔧"
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

        print(f"{emoji} {message.content}")


async def main():

    client = MultiServerMCPClient(
        {
            "similarity_search": {
                "url": "http://51.77.212.235:9000/mcp",
                "transport": "streamable_http",
            },
            "add_repo_to_vector_store": {
                "url": "http://51.77.212.235:9000/mcp",
                "transport": "streamable_http",
            },
            "delete_repo_from_vector_store": {
                "url": "http://51.77.212.235:9000/mcp",
                "transport": "streamable_http",
            },
        }
    )

    tools = await client.get_tools()

    agent = create_react_agent(
        model="openai:gpt-4.1",
        prompt="You are a helpful assistant that is specifically designed to answer questions about a codebase, the content is stored in a vector store. Please only answer questions about the codebase by running the similarity_search tool.",
        tools=tools,
    )
    response = await agent.ainvoke(
        {
            # "messages": "is there any java code in torchjd?"
            # "messages": "can you remove TorchJD/torchjd from the vector store?"
            "messages": "can you explain the implementation of UPGrad?"
        }
    )
    print_messages(response)


if __name__ == "__main__":
    asyncio.run(main())
