import asyncio

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

load_dotenv()


async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["mathserver.py"],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            },
        }
    )

    tools = await client.get_tools()

    model = ChatGroq(model="qwen/qwen3.8-27b")

    agent = create_agent(
        model=model,
        tools=tools,
    )

    # Math
    math_response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "what's (3+5)x12?",
                }
            ]
        }
    )

    print("Math's Response:", math_response["messages"][-1].content)

    # Weather
    weather_response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "what is the weather in California?",
                }
            ]
        }
    )

    print("Weather Response:", weather_response["messages"][-1].content)


asyncio.run(main())
