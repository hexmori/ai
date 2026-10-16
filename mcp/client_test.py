# client_test.py
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_mcp_server():
    """测试MCP服务器的各种功能"""

    # 创建客户端连接到服务器
    server_params = StdioServerParameters(
        command="python",
        args=["weather_server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化会话
            await session.initialize()

            print("=" * 60)
            print("🚀 开始测试MCP服务器")
            print("=" * 60)

            # ===== 测试1: 列出资源 =====
            print("\n📚 测试1: 列出所有资源")
            resources = await session.list_resources()
            for resource in resources.resources:
                print(f"  - {resource.name}: {resource.uri}")

            # ===== 测试2: 读取资源 =====
            print("\n📖 测试2: 读取资源内容")
            cities_data = await session.read_resource("weather://cities")
            print(f"  城市列表: {cities_data.contents[0].text}")

            # ===== 测试3: 列出工具 =====
            print("\n🔧 测试3: 列出所有工具")
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")

            # ===== 测试4: 调用工具 - 查询天气 =====
            print("\n🌤️  测试4: 调用工具查询天气")
            result = await session.call_tool(
                "get_weather",
                arguments={"city": "北京", "day": "today"}
            )
            print(result.content[0].text)

            # ===== 测试5: 调用工具 - 比较天气 =====
            print("\n🔍 测试5: 调用工具比较天气")
            result = await session.call_tool(
                "compare_weather",
                arguments={"city1": "北京", "city2": "广州"}
            )
            print(result.content[0].text)

            # ===== 测试6: 调用工具 - 设置预警 =====
            print("\n⚠️  测试6: 调用工具设置预警")
            result = await session.call_tool(
                "add_city_alert",
                arguments={"city": "上海", "condition": "温度超过30度"}
            )
            print(result.content[0].text)

            # ===== 测试7: 获取提示词 =====
            print("\n💬 测试7: 获取提示词模板")
            prompt = await session.get_prompt(
                "weather_analysis",
                arguments={"city": "北京"}
            )
            print(f"  提示词: {prompt.messages[0].content.text[:100]}...")

            print("\n" + "=" * 60)
            print("✅ 所有测试完成！")


if __name__ == "__main__":
    asyncio.run(test_mcp_server())