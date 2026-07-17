# afm_agent.py
import os
import json
from openai import OpenAI

# 1. 初始化 DeepSeek 客户端
client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com/v1",
    timeout=30.0,
    max_retries=2,
)

# 2. 定义工具 (Tools) - 修复了 get_simulation_status 的 parameters 格式
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_hysteresis_simulation",
            "description": "运行一次AFM迟滞效应模拟，并返回模拟结果。",
            "parameters": {
                "type": "object",
                "properties": {
                    "scan_range": {
                        "type": "number",
                        "description": "扫描范围（单位：微米）"
                    },
                    "scan_speed": {
                        "type": "number",
                        "description": "扫描速度（单位：微米/秒）"
                    }
                },
                "required": ["scan_range", "scan_speed"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_rotation_angle",
            "description": "根据AI识别的图像，调整模拟中的样品旋转角度，以修正倾斜。",
            "parameters": {
                "type": "object",
                "properties": {
                    "angle": {
                        "type": "number",
                        "description": "需要调整的旋转角度（单位：度），范围在-10到10之间。"
                    }
                },
                "required": ["angle"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_simulation_status",
            "description": "获取当前模拟的状态信息，如运行进度、当前参数等。",
            "parameters": {
                "type": "object",
                "properties": {}   # 无参数，但必须明确声明 type: object
            }
        }
    }
]

# 3. 实现工具的执行逻辑
def execute_tool_call(tool_call):
    """根据AI的指令，执行对应的工具函数"""
    tool_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    if tool_name == "run_hysteresis_simulation":
        # 🔧 这里替换成你项目中实际的模拟函数
        print(f"-> 执行模拟: 范围={arguments['scan_range']}um, 速度={arguments['scan_speed']}um/s")
        return f"模拟完成。结果: 扫描范围 {arguments['scan_range']}um，速度 {arguments['scan_speed']}um/s，迟滞效应已模拟。"
    
    elif tool_name == "adjust_rotation_angle":
        # 🔧 这里调用你项目中调整旋转角度的函数
        print(f"-> 调整旋转角度: {arguments['angle']}度")
        return f"旋转角度已调整为 {arguments['angle']} 度。"
    
    elif tool_name == "get_simulation_status":
        # 🔧 这里获取你项目中的模拟状态
        return "当前模拟状态: 空闲中，等待指令。"
    
    return f"未知工具: {tool_name}"

# 4. Agent 主循环
def run_agent(user_query):
    print(f"用户: {user_query}\n")
    print(f"正在使用 API 端点: {client.base_url}\n")

    messages = [
        {"role": "system", "content": "你是一个AFM模拟助手，可以帮助用户运行模拟、调整参数。你必须通过调用提供的工具来完成任务。"},
        {"role": "user", "content": user_query}
    ]

    try:
        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )
    except Exception as e:
        print(f"❌ API 调用失败: {e}")
        return

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        print(f"🤖 AI 决定调用工具: {tool_calls[0].function.name}")
        
        for tool_call in tool_calls:
            function_result = execute_tool_call(tool_call)
            messages.append(response_message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": function_result
            })
        
        try:
            second_response = client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=messages,
            )
            print(f"\n🤖 AI 最终回复: {second_response.choices[0].message.content}")
        except Exception as e:
            print(f"❌ 第二次 API 调用失败: {e}")
    else:
        print(f"🤖 AI 回复: {response_message.content}")

# 5. 测试入口
if __name__ == "__main__":
    if not os.environ.get('DEEPSEEK_API_KEY'):
        print("⚠️ 请先设置环境变量 DEEPSEEK_API_KEY")
        print("在 PowerShell 中执行: $env:DEEPSEEK_API_KEY='你的API密钥'")
    else:
        run_agent("请帮我运行一次扫描范围为5微米，速度为2微米/秒的模拟。")
        print("\n" + "="*60 + "\n")
        run_agent("当前的模拟状态是什么？")