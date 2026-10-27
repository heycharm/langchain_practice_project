import asyncio

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

# Load environment variables from the .env file
# This is where your Groq API key and other secrets can be stored.
load_dotenv()


async def main():

    # Create an MCP client that can connect to multiple MCP servers.
    #
    # Here we are connecting to two different MCP servers:
    # 1. Math MCP server using stdio
    # 2. Weather MCP server using Streamable HTTP
    client = MultiServerMCPClient(
        {
            # Math MCP server
            "math": {
                # Command used to start the MCP server.
                # The MCP client will run this Python process.
                "command": "python",
                # Python file that contains the math MCP server.
                "args": ["mathserver.py"],
                # stdio means communication happens through
                # the server's standard input and output.
                "transport": "stdio",
            },
            # Weather MCP server
            "weather": {
                # URL of the already-running MCP server.
                "url": "http://localhost:8000/mcp",
                # Streamable HTTP allows the MCP client to communicate
                # with the server through HTTP.
                "transport": "streamable_http",
            },
        }
    )

    # Fetch all tools exposed by both MCP servers.
    #
    # The returned list will contain the tools from:
    # - mathserver.py
    # - weather MCP server
    tools = await client.get_tools()

    # Create the LLM that will act as the agent's brain.
    #
    # The agent will decide which MCP tool should be used
    # based on the user's request.
    model = ChatGroq(model="qwen/qwen3.8-27b")

    # Create a LangChain agent and provide it with:
    # - The Groq model
    # - All tools retrieved from the MCP servers
    #
    # The agent can now decide when to call the math or weather tools.
    agent = create_agent(
        model=model,
        tools=tools,
    )

    # -------------------------
    # Math MCP Server Test
    # -------------------------

    # Ask the agent to calculate a mathematical expression.
    #
    # The agent should recognize that this requires the
    # math MCP server and call the appropriate math tool.
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

    # The last message contains the agent's final response.
    print("Math's Response:", math_response["messages"][-1].content)

    # -------------------------
    # Weather MCP Server Test
    # -------------------------

    # Ask the agent for weather information.
    #
    # The agent should recognize that this requires the
    # weather MCP server and call the appropriate weather tool.
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

    # Print the agent's final weather response.
    print("Weather Response:", weather_response["messages"][-1].content)


# Start the asynchronous main function.
asyncio.run(main())
