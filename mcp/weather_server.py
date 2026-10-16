# weather_server.py
import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# 模拟天气数据存储
WEATHER_DATA = {
    "北京": {
        "today": {"temp": 28, "condition": "晴", "humidity": 45, "wind": "3级"},
        "tomorrow": {"temp": 30, "condition": "多云", "humidity": 50, "wind": "4级"},
        "city_code": "beijing"
    },
    "上海": {
        "today": {"temp": 32, "condition": "小雨", "humidity": 78, "wind": "2级"},
        "tomorrow": {"temp": 31, "condition": "中雨", "humidity": 82, "wind": "3级"},
        "city_code": "shanghai"
    },
    "广州": {
        "today": {"temp": 35, "condition": "雷阵雨", "humidity": 85, "wind": "3级"},
        "tomorrow": {"temp": 34, "condition": "多云", "humidity": 75, "wind": "2级"},
        "city_code": "guangzhou"
    }
}

# 创建MCP服务器实例
server = Server("weather-server")


# ============ 定义资源 (Resources) ============
# 资源提供静态或动态数据，供AI读取

@server.list_resources()
async def handle_list_resources() -> List[types.Resource]:
    """列出所有可用资源"""
    return [
        types.Resource(
            uri=f"weather://cities",
            name="所有支持的城市列表",
            description="获取所有支持查询天气的城市及其代码",
            mime_type="application/json",
        ),
        types.Resource(
            uri=f"weather://city/{city}",
            name=f"{city}的天气信息",
            description=f"获取{city}的实时天气状况",
            mime_type="application/json",
        )
    # for city in WEATHER_DATA.keys()
    ]


@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """读取资源内容"""
    # 处理城市列表请求
    if uri == "weather://cities":
        cities = [
            {"name": city, "code": info["city_code"]}
            for city, info in WEATHER_DATA.items()
        ]
        return json.dumps(cities, ensure_ascii=False, indent=2)

    # 处理特定城市天气请求
    if uri.startswith("weather://city/"):
        city = uri.replace("weather://city/", "")
        if city in WEATHER_DATA:
            # 添加更新时间
            result = {
                "city": city,
                "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **WEATHER_DATA[city]
            }
            return json.dumps(result, ensure_ascii=False, indent=2)

    raise ValueError(f"未知资源URI: {uri}")


# ============ 定义工具 (Tools) ============
# 工具是AI可以调用的可执行函数

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """列出所有可用工具"""
    return [
        types.Tool(
            name="get_weather",
            description="获取指定城市的天气预报",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：北京、上海、广州",
                        "enum": list(WEATHER_DATA.keys())
                    },
                    "day": {
                        "type": "string",
                        "description": "查询哪天的天气：today 或 tomorrow",
                        "enum": ["today", "tomorrow"],
                        "default": "today"
                    }
                },
                "required": ["city"]
            }
        ),
        types.Tool(
            name="compare_weather",
            description="比较两个城市的天气情况",
            inputSchema={
                "type": "object",
                "properties": {
                    "city1": {
                        "type": "string",
                        "description": "第一个城市名称",
                        "enum": list(WEATHER_DATA.keys())
                    },
                    "city2": {
                        "type": "string",
                        "description": "第二个城市名称",
                        "enum": list(WEATHER_DATA.keys())
                    }
                },
                "required": ["city1", "city2"]
            }
        ),
        types.Tool(
            name="add_city_alert",
            description="为某个城市设置天气预警",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称",
                        "enum": list(WEATHER_DATA.keys())
                    },
                    "condition": {
                        "type": "string",
                        "description": "预警条件，如：温度超过30度，湿度高于80%等"
                    }
                },
                "required": ["city", "condition"]
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(
        name: str,
        arguments: Dict[str, Any] | None
) -> List[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """执行工具调用"""

    if not arguments:
        arguments = {}

    # 工具1: 获取天气
    if name == "get_weather":
        city = arguments.get("city")
        day = arguments.get("day", "today")

        if city not in WEATHER_DATA:
            return [types.TextContent(
                type="text",
                text=f"❌ 不支持的城市: {city}，支持的城市有: {', '.join(WEATHER_DATA.keys())}"
            )]

        city_data = WEATHER_DATA[city]
        if day not in city_data:
            return [types.TextContent(
                type="text",
                text=f"❌ 不支持查询 {day}，支持: today, tomorrow"
            )]

        weather = city_data[day]
        result = f"""
📍 {city} {day}天气预报:
🌡️  温度: {weather['temp']}°C
☁️  天气: {weather['condition']}
💧  湿度: {weather['humidity']}%
💨  风力: {weather['wind']}
📅  更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return [types.TextContent(type="text", text=result.strip())]

    # 工具2: 比较天气
    elif name == "compare_weather":
        city1 = arguments.get("city1")
        city2 = arguments.get("city2")

        if city1 not in WEATHER_DATA or city2 not in WEATHER_DATA:
            return [types.TextContent(
                type="text",
                text="❌ 请输入正确的城市名称"
            )]

        w1 = WEATHER_DATA[city1]["today"]
        w2 = WEATHER_DATA[city2]["today"]

        # 计算温度差和湿度差
        temp_diff = abs(w1['temp'] - w2['temp'])
        humidity_diff = abs(w1['humidity'] - w2['humidity'])

        result = f"""
🌆 天气对比结果:

┌─────────────┬─────────────┐
│    {city1:<8} │    {city2:<8} │
├─────────────┼─────────────┤
│ 🌡️ {w1['temp']}°C    │ 🌡️ {w2['temp']}°C    │
│ ☁️ {w1['condition']:<6} │ ☁️ {w2['condition']:<6} │
│ 💧 {w1['humidity']}%    │ 💧 {w2['humidity']}%    │
│ 💨 {w1['wind']:<6} │ 💨 {w2['wind']:<6} │
└─────────────┴─────────────┘

📊 分析:
• 温度相差 {temp_diff}°C，{city1 if w1['temp'] > w2['temp'] else city2} 更热
• 湿度相差 {humidity_diff}%
"""
        return [types.TextContent(type="text", text=result.strip())]

    # 工具3: 设置预警
    elif name == "add_city_alert":
        city = arguments.get("city")
        condition = arguments.get("condition", "")

        # 模拟存储预警
        alert_id = f"alert_{datetime.now().timestamp()}"
        result = f"""
✅ 预警设置成功！

城市: {city}
预警条件: {condition}
预警ID: {alert_id}
状态: 已激活
创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

💡 当 {city} 的天气满足条件 "{condition}" 时，系统将发送通知。
"""
        return [types.TextContent(type="text", text=result.strip())]

    else:
        raise ValueError(f"未知工具: {name}")


# ============ 可选：定义提示词 (Prompts) ============
@server.list_prompts()
async def handle_list_prompts() -> List[types.Prompt]:
    """列出可用的提示词模板"""
    return [
        types.Prompt(
            name="weather_analysis",
            description="生成一个城市的天气分析报告",
            arguments=[
                types.PromptArgument(
                    name="city",
                    description="要分析的城市名称",
                    required=True
                )
            ]
        )
    ]


@server.get_prompt()
async def handle_get_prompt(
        name: str,
        arguments: Dict[str, str] | None
) -> types.GetPromptResult:
    """获取特定提示词模板"""
    if name == "weather_analysis" and arguments:
        city = arguments.get("city", "")
        if city in WEATHER_DATA:
            weather = WEATHER_DATA[city]["today"]
            return types.GetPromptResult(
                messages=[
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(
                            type="text",
                            text=f"请根据以下数据生成一份关于{city}的天气分析报告：\n"
                                 f"温度{weather['temp']}°C，{weather['condition']}，"
                                 f"湿度{weather['humidity']}%，{weather['wind']}。"
                                 f"请包含出行建议和穿衣指南。"
                        )
                    )
                ]
            )

    raise ValueError(f"未知提示词: {name}")


# ============ 启动服务器 ============
async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="weather-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())