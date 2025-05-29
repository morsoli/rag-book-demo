# 导入必要的库
from typing import List, Dict, Any
import math
import json
import random
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain.llms import OpenAI  # 模拟LLM，实际需替换为真实模型
import warnings
warnings.filterwarnings("ignore")  # 忽略警告

# 模拟大语言模型（LLM）输出
class MockLLM:
    def __init__(self):
        self.responses = {
            # 零样本情感分类
            "对以下文本进行情感分类，可选项为中性、消极或积极。\n文本：大模型的效果还不错。\n情感：": "积极",
            # 少样本造句
            '"滴溜"是一种描述物体快速旋转的拟声词。使用"滴溜"造句：小明把陀螺一甩，它就滴溜滴溜地转了起来。\n"咕噜"是形容液体冒泡或人肚子叫的声音。使用"咕噜"造句：': "热水壶里的水煮沸了，咕噜咕噜地冒着泡。",
            # 少样本情感分析（随机标签）
            '这部电影太无聊了！ // 积极\n今天的天气真好啊！ // 消极\n我刚刚丢了钱包。 // 积极\n这家餐厅的食物很美味。 // ': "积极",
            # 少样本数学推理（草莓问题）
            '计算每个单词有多少个字母？\n问题1：apple有多少个p？\n答案：2\n\n问题2：language有多少个a?\n答案：2\n\n问题3：strawberry有多少个r？\n答案：': "3",  # 修正为正确答案
            # 思维链数学推理
            '问题：一个农场有15只鸡，12只鸭子和8只鹅。如果每只鸡每天下2个蛋，每只鸭子每天下3个蛋，每只鹅每天下1个蛋，那么一周内农场总共会有多少个蛋？\n\n思考步骤：\n1. 计算鸡每天下的蛋数：15 * 2 = 30个\n2. 计算鸭子每天下的蛋数：12 * 3 = 36个\n3. 计算鹅每天下的蛋数：8 * 1 = 8个\n4. 计算每天总共下的蛋数：30 + 36 + 8 = 74个\n5. 计算一周内的总蛋数：74 * 7 = 518个\n\n因此，一周内农场总共会有518个蛋。\n\n新问题：一个商店有50件T恤，每件售价15元。如果周一卖出10件，周二卖出15件，周三打八折促销并卖出剩下的一半，那么到周三结束时，商店的总收入是多少？': """思考步骤：
1. 计算周一的销售收入：10 * 15 = 150元
2. 计算周二的销售收入：15 * 15 = 225元
3. 计算周三之前剩余的T恤数量：50 - 10 - 15 = 25件
4. 计算周三卖出的T恤数量：25 / 2 = 12.5件（向下取整为12件）
5. 计算周三的折扣价格：15 * 0.8 = 12元
6. 计算周三的销售收入：12 * 12 = 144元
7. 计算总收入：150 + 225 + 144 = 519元

因此，到周三结束时，商店的总收入是519元。""",
            # 零样本CoT
            '一个家庭决定种植一个小花园。他们买了3包向日葵种子，每包有20颗种子。如果他们种下了总数的四分之三，并且其中85%成功发芽，那么最终有多少株向日葵长出来了？Let\'s think it step by step.': """让我们逐步思考这个问题：

1. 计算总种子数：
   3包 * 20颗/包 = 60颗种子

2. 计算种下的种子数（四分之三）：
   60颗 * 3/4 = 45颗种子

3. 计算成功发芽的种子数（85%）：
   45颗 * 85% = 45 * 0.85 = 38.25颗

4. 由于我们不能有0.25株向日葵，所以我们向下取整：
   38株向日葵

因此，最终有38株向日葵长出来了。"""
        }

    def __call__(self, prompt: str) -> str:
        """模拟LLM调用"""
        for key, value in self.responses.items():
            if prompt.strip().endswith(key.strip()):
                return value
        return "未找到匹配的响应"

# === 零样本提示 ===
def zero_shot_prompt(text: str) -> str:
    """零样本情感分类"""
    prompt = f"""对以下文本进行情感分类，可选项为中性、消极或积极。
文本：{text}
情感："""
    llm = MockLLM()
    return llm(prompt)

# === 少样本提示 ===
def few_shot_sentence_making(word: str, description: str, example: Dict[str, str]) -> str:
    """少样本造句"""
    example_word, example_sentence = list(example.items())[0]
    prompt = f'''"{example_word}"是{example["description"]}。使用"{example_word}"造句：{example_sentence}
"{word}"是{description}。使用"{word}"造句：'''
    llm = MockLLM()
    return llm(prompt)

def few_shot_sentiment_analysis(text: str, examples: List[Dict[str, str]]) -> str:
    """少样本情感分析（随机标签）"""
    prompt = ""
    for ex in examples:
        prompt += f"{ex['text']} // {ex['label']}\n"
    prompt += f"{text} // "
    llm = MockLLM()
    return llm(prompt)

def few_shot_math_counting(word: str, letter: str, examples: List[Dict[str, Any]]) -> str:
    """少样本数学推理（计数问题）"""
    prompt = "计算每个单词有多少个字母？\n"
    for ex in examples:
        prompt += f"问题：{ex['word']}有多少个{ex['letter']}？\n答案：{ex['count']}\n\n"
    prompt += f"问题：{word}有多少个{letter}？\n答案："
    llm = MockLLM()
    return llm(prompt)

# === 思维链提示 ===
def chain_of_thought_prompt(question: str, example: Dict[str, Any]) -> str:
    """思维链提示"""
    example_question = example["question"]
    example_steps = example["steps"]
    example_answer = example["answer"]
    prompt = f"""问题：{example_question}

思考步骤：
{example_steps}

因此，{example_answer}

新问题：{question}"""
    llm = MockLLM()
    return llm(prompt)

# === 零样本思维链提示 ===
def zero_shot_cot_prompt(question: str) -> str:
    """零样本思维链提示"""
    prompt = f"{question} Let's think it step by step."
    llm = MockLLM()
    return llm(prompt)

# === ReAct 框架 ===
class ReActAgent:
    def __init__(self, tools: List[Dict[str, str]]):
        self.tools = tools
        self.tool_names = [tool["name"] for tool in tools]
        self.llm = MockLLM()
        self.max_iterations = 5

    def run(self, question: str) -> str:
        """运行ReAct代理"""
        prompt = f"""Answer the following questions as best you can. You have access to the following tools:
{self.tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of {self.tool_names}
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {question}
Thought:"""
        
        # 模拟工具调用
        history = [prompt]
        iteration = 0
        while iteration < self.max_iterations:
            response = self.llm("\n".join(history))
            history.append(response)
            
            if "Final Answer" in response:
                return response
            
            # 模拟工具执行
            if "Action:" in response:
                action = response.split("Action:")[1].split("\n")[0].strip()
                action_input = response.split("Action Input:")[1].split("\n")[0].strip()
                observation = self.execute_tool(action, action_input)
                history.append(f"Observation: {observation}")
                history.append("Thought:")
            
            iteration += 1
        
        return "未能在最大迭代次数内完成"

    def execute_tool(self, action: str, action_input: str) -> str:
        """模拟工具执行"""
        if action == "Weather":
            return f"The current weather in {action_input} is partly cloudy with a temperature of 22°C (72°F)."
        elif action == "Calculator":
            try:
                if action_input.startswith("sqrt"):
                    num = float(action_input.replace("sqrt(", "").replace(")", ""))
                    return str(math.sqrt(num))
                return str(eval(action_input))
            except:
                return "计算错误"
        elif action == "Search":
            return f"搜索结果：关于{action_input}的最新信息。"
        return "未知工具"

# 主函数：演示提示工程技术
if __name__ == "__main__":
    # === 零样本提示 ===
    print("=== 零样本提示 ===")
    text = "大模型的效果还不错。"
    sentiment = zero_shot_prompt(text)
    print(f"文本：{text}\n情感：{sentiment}")

    # === 少样本提示 ===
    print("\n=== 少样本提示：造句 ===")
    word = "咕噜"
    description = "形容液体冒泡或人肚子叫的声音"
    example = {
        "滴溜": {
            "description": "一种描述物体快速旋转的拟声词",
            "sentence": "小明把陀螺一甩，它就滴溜滴溜地转了起来。"
        }
    }
    sentence = few_shot_sentence_making(word, description, example)
    print(f"单词：{word}\n造句：{sentence}")

    print("\n=== 少样本提示：情感分析 ===")
    text = "这家餐厅的食物很美味。"
    examples = [
        {"text": "这部电影太无聊了！", "label": "积极"},
        {"text": "今天的天气真好啊！", "label": "消极"},
        {"text": "我刚刚丢了钱包。", "label": "积极"}
    ]
    sentiment = few_shot_sentiment_analysis(text, examples)
    print(f"文本：{text}\n情感：{sentiment}")

    print("\n=== 少样本提示：数学推理 ===")
    word = "strawberry"
    letter = "r"
    examples = [
        {"word": "apple", "letter": "p", "count": 2},
        {"word": "language", "letter": "a", "count": 2}
    ]
    count = few_shot_math_counting(word, letter, examples)
    print(f"问题：{word}有多少个{letter}？\n答案：{count}")

    # === 思维链提示 ===
    print("\n=== 思维链提示 ===")
    question = "一个商店有50件T恤，每件售价15元。如果周一卖出10件，周二卖出15件，周三打八折促销并卖出剩下的一半，那么到周三结束时，商店的总收入是多少？"
    example = {
        "question": "一个农场有15只鸡，12只鸭子和8只鹅。如果每只鸡每天下2个蛋，每只鸭子每天下3个蛋，每只鹅每天下1个蛋，那么一周内农场总共会有多少个蛋？",
        "steps": """1. 计算鸡每天下的蛋数：15 * 2 = 30个
2. 计算鸭子每天下的蛋数：12 * 3 = 36个
3. 计算鹅每天下的蛋数：8 * 1 = 8个
4. 计算每天总共下的蛋数：30 + 36 + 8 = 74个
5. 计算一周内的总蛋数：74 * 7 = 518个""",
        "answer": "一周内农场总共会有518个蛋。"
    }
    answer = chain_of_thought_prompt(question, example)
    print(f"问题：{question}\n回答：\n{answer}")

    # === 零样本思维链提示 ===
    print("\n=== 零样本思维链提示 ===")
    question = "一个家庭决定种植一个小花园。他们买了3包向日葵种子，每包有20颗种子。如果他们种下了总数的四分之三，并且其中85%成功发芽，那么最终有多少株向日葵长出来了？"
    answer = zero_shot_cot_prompt(question)
    print(f"问题：{question}\n回答：\n{answer}")

    # === ReAct 框架 ===
    print("\n=== ReAct 框架 ===")
    tools = [
        {"name": "Search", "description": "useful for when you need to answer questions about current events"},
        {"name": "Calculator", "description": "useful for when you need to perform mathematical calculations"},
        {"name": "Weather", "description": "useful for when you need to check the weather in a specific location"}
    ]
    question = "今天纽约的天气如何，以及当前温度（摄氏度）的平方根是多少？"
    agent = ReActAgent(tools)
    response = agent.run(question)
    print(f"问题：{question}\n响应：\n{response}")
