from aiohttp import web
import json

async def start_web_interface(bot):
    app = web.Application()
    app['bot'] = bot

    async def status_handler(request):
        bot = request.app['bot']
        status = {
            'running': bot.running,
            'open_trades': bot.portfolio_manager.get_open_positions(),
            'total_exposure': bot.portfolio_manager.get_total_exposure(),
        }
        return web.json_response(status)

    async def start_handler(request):
        bot = request.app['bot']
        if not bot.running:
            await bot.run()
            return web.json_response({'status': 'Bot started'})
        else:
            return web.json_response({'status': 'Bot is already running'})

    async def stop_handler(request):
        bot = request.app['bot']
        if bot.running:
            bot.stop()
            return web.json_response({'status': 'Bot stopped'})
        else:
            return web.json_response({'status': 'Bot is not running'})

    async def backtest_handler(request):
        bot = request.app['bot']
        results = await bot.run_backtest()
        return web.json_response(results)

    app.router.add_get('/status', status_handler)
    app.router.add_post('/start', start_handler)
    app.router.add_post('/stop', stop_handler)
    app.router.add_post('/backtest', backtest_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8080)
    await site.start()

    print(f"Web interface started at http://localhost:8080")

