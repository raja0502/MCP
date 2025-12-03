from mcp.client import StdioClient
import asyncio
import json

async def main():
    client = StdioClient(command=["python", "dq_mcp_server.py"])
    await client.start()

    # Call profile_table
    result1 = await client.call_tool("profile_table", {"table_name": "customers"})
    print("PROFILE RESULT:")
    print(json.dumps(result1, indent=2))

    # Call check_negative_balance
    result2 = await client.call_tool("check_negative_balance", {"table_name": "customers"})
    print("\nNEGATIVE BALANCE RESULT:")
    print(json.dumps(result2, indent=2))

    await client.stop()

if __name__ == "__main__":
    asyncio.run(main())
