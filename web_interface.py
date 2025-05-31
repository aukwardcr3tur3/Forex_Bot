from aiohttp import web
import json
import logging
import reactivex as rx
from reactivex import operators as ops
from bot import ForexBot

BOT_KEY = web.AppKey("bot")

async def start_web_interface(bot: ForexBot):
    app = web.Application()
    app[BOT_KEY] = bot
    logging.getLogger('aiohttp').setLevel(logging.INFO)

    async def get_bot_status():
        try:
            return {
                "running": bot.running,
                "open_trades": bot.portfolio_manager.get_open_positions(),
                "total_exposure": bot.portfolio_manager.get_total_exposure(),
            }
        except Exception as e:
            logging.error("Error getting bot status: %s", str(e))
            return {"error": "Failed to retrieve bot status"}

    async def status_handler(request):
        status = await get_bot_status()
        return web.json_response(status)

    async def start_handler(request):
        if not bot.running:
            asyncio.create_task(bot.run())
            return web.json_response({"status": "Bot started"})
        return web.json_response({"status": "Bot is already running"})

    async def stop_handler(request):
        if bot.running:
            await bot.stop()
            return web.json_response({"status": "Bot stopped"})
        return web.json_response({"status": "Bot is not running"})

    async def backtest_handler(request):
        try:
            results = await bot.run_backtest()
            source = rx.just(results)
            source.subscribe(
                lambda r: logging.info(f"Backtest results streamed: {r}"),
                lambda e: logging.error(f"Backtest stream error: {e}")
            )
            return web.json_response(results)
        except Exception as e:
            logging.error("Error during backtest: %s", str(e))
            return web.json_response({"error": "Backtest failed"}, status=500)

    async def websocket_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        try:
            status = await get_bot_status()
            await ws.send_json({"type": "status", "data": status})
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get("action") == "get_status":
                        status = await get_bot_status()
                        await ws.send_json({"type": "status", "data": status})
                elif msg.type == web.WSMsgType.ERROR:
                    logging.error(f"WebSocket error: {ws.exception()}")
        except Exception as e:
            logging.error(f"WebSocket handler error: {e}")
        return ws

    app.router.add_get("/status", status_handler)
    app.router.add_post("/start", start_handler)
    app.router.add_post("/stop", stop_handler)
    app.router.add_post("/backtest", backtest_handler)
    app.router.add_get("/ws", websocket_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", 8080)
    await site.start()

    logging.info("Web interface started at http://localhost:8080")
    return runner
