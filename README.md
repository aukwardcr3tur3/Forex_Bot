# Forex_Bot
An advanced forex trading bot that incorporates risk management techniques, error handling, continuous monitoring, consideration of transaction costs, trailing stops, slippage handling, and careful consideration of factors like market data latency and order execution quality using api.deriv.com.
1. Ensure you have Python 3.7 or higher installed on your system.

2. Create a new directory for your project and navigate to it in the terminal:

3. mkdir advanced_forex_bot

4. cd advanced_forex_bot

5. Create a virtual environment:
   python -m venv stonks

6. Activate the virtual environment:

   On Windows: venv\Scripts\activate
   On macOS and Linux: source venv/bin/activate
  
8. Install the required packages:

   pip install deriv-api python-dotenv numpy pandas scikit-learn TA-Lib aiohttp

   Note: Installing TA-Lib might be tricky on some systems. You may need to install it separately following the 
   instructions for your operating system.

9. Create a .env file in the project directory and add your Deriv API token:

DERIV_API_TOKEN=
APP_ID=

10. DERIV_API_TOKEN=your_api_token_here

11. Run the bot:
   python3 Forex-Main.py

   The bot will start running, and you can access the web interface at http://localhost:8080.

   #Checkout Arbitagelab (https://hudsonthames.org/arbitragelab/)
   #Checkout GeneticAlgos(https://slicematrix.github.io/python-docs/)

11. To interact with the bot through the web interface, you can use curl commands or any HTTP client:

   Get bot status: curl http://localhost:8080/status
   Start the bot: curl -X POST http://localhost:8080/start
   Stop the bot: curl -X POST http://localhost:8080/stop
   Run backtest: curl -X POST http://localhost:8080/backtest

   Remember to replace your_api_token_here in the .env file with your actual Deriv API token.

   Disclaimer
This bot is for educational purposes only. Trading forex carries a high level of risk and may not be suitable for all investors. Please ensure you fully understand the risks involved and seek independent advice if necessary.

Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

License
This project is licensed under the MIT License.
