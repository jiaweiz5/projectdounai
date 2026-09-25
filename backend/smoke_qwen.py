import asyncio

from qwen_client import ask_qwen


async def main():
    print("Sending a request to Qwen...")

    answer = await ask_qwen([
        {
            "role": "user",
            "content": "请只回复：QWEN_OK：中文连接成功"
        }
    ])

    print("\nQwen response:")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())