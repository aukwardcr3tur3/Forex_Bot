import sys,time
import asyncio
import websockets
from bot import ForexBot
from config import Config
from web_interface import start_web_interface

async def main():
    config = Config()
    if config.API_TOKEN is None or len(config.API_TOKEN) == 0:
        print("DERIV_TOKEN environment variable is not set")
        time.sleep(2)
        sys.exit("Exiting....")

    connection = await websockets.connect(config.EndPoint)
    bot = ForexBot(config, connection)

    
    web_task = asyncio.create_task(start_web_interface(bot))
    bot_task = asyncio.create_task(bot.run())
    
    try:
        await asyncio.gather(web_task, bot_task)
    except KeyboardInterrupt:
        print("Shutting down gracefully...")
    # Perform any cleanup here if necessary

if __name__ == "__main__":
    asyncio.run(main())

