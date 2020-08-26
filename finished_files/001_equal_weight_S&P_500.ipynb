{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Equal-Weight S&P 500 Index Fund\n",
    "\n",
    "## Introduction & Library Imports\n",
    "\n",
    "The S&P 500 is the world's most popular stock market index. The largest fund that is benchmarked to this index is the SPDR® S&P 500® ETF Trust. It has more than US$250 billion of assets under management.\n",
    "\n",
    "The goal of this section of the course is to create a Python script that will accept the value of your portfolio and tell you how many shares of each S&P 500 constituent you should purchase to get an equal-weight version of the index fund.\n",
    "\n",
    "## Library Imports\n",
    "\n",
    "The first thing we need to do is import the open-source software libraries that we'll be using in this tutorial."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np #The Numpy numerical computing library\n",
    "import pandas as pd #The Pandas data science library\n",
    "import requests #The requests library for HTTP requests in Python\n",
    "import xlsxwriter #The XlsxWriter libarary for \n",
    "import math #The Python math module"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Importing Our List of Stocks\n",
    "\n",
    "The next thing we need to do is import the constituents of the S&P 500.\n",
    "\n",
    "These constituents change over time, so in an ideal world you would connect directly to the index provider (Standard & Poor's) and pull their real-time constituents on a regular basis.\n",
    "\n",
    "Paying for access to the index provider's API is outside of the scope of this course. \n",
    "\n",
    "There's a static version of the S&P 500 constituents available here. [Click this link to download them now](http://nickmccullum.com/algorithmic-trading-python/sp_500_stocks.csv). Move this file into the `starter-files` folder so it can be accessed by other files in that directory.\n",
    "\n",
    "Now it's time to import these stocks to our Jupyter Notebook file."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "stocks = pd.read_csv('sp_500_stocks.csv')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Acquiring an API Token\n",
    "\n",
    "Now it's time to import our IEX Cloud API token. This is the data provider that we will be using throughout this course.\n",
    "\n",
    "API tokens (and other sensitive information) should be stored in a `secrets.py` file that doesn't get pushed to your local Git repository. We'll be using a sandbox API token in this course, which means that the data we'll use is randomly-generated and (more importantly) has no cost associated with it.\n",
    "\n",
    "[Click here](http://nickmccullum.com/algorithmic-trading-python/secrets.py) to download your `secrets.py` file. Move the file into the same directory as this Jupyter Notebook before proceeding."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from secrets import IEX_CLOUD_API_TOKEN"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Making Our First API Call\n",
    "\n",
    "Now it's time to structure our API calls to IEX cloud. \n",
    "\n",
    "We need the following information from the API:\n",
    "\n",
    "* Market capitalization for each stock\n",
    "* Price of each stock\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "symbol='AAPL'\n",
    "api_url = f'https://sandbox.iexapis.com/stable/stock/{symbol}/quote?token={IEX_CLOUD_API_TOKEN}'\n",
    "data = requests.get(api_url).json()\n",
    "data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Parsing Our API Call\n",
    "\n",
    "The API call that we executed in the last code block contains all of the information required to build our equal-weight S&P 500 strategy. \n",
    "\n",
    "With that said, the data isn't in a proper format yet. We need to parse it first."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['latestPrice']\n",
    "data['marketCap']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Adding Our Stocks Data to a Pandas DataFrame\n",
    "\n",
    "The next thing we need to do is add our stock's price and market capitalization to a pandas DataFrame. Think of a DataFrame like the Python version of a spreadsheet. It stores tabular data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "my_columns = ['Ticker', 'Price','Market Capitalization', 'Number Of Shares to Buy']\n",
    "final_dataframe = pd.DataFrame(columns = my_columns)\n",
    "final_dataframe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "final_dataframe = final_dataframe.append(\n",
    "                                        pd.Series(['AAPL', \n",
    "                                                   data['latestPrice'], \n",
    "                                                   data['marketCap'], \n",
    "                                                   'N/A'], \n",
    "                                                  index = my_columns), \n",
    "                                        ignore_index = True)\n",
    "final_dataframe"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Looping Through The Tickers in Our List of Stocks\n",
    "\n",
    "Using the same logic that we outlined above, we can pull data for all S&P 500 stocks and store their data in the DataFrame using a `for` loop."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "final_dataframe = pd.DataFrame(columns = my_columns)\n",
    "for symbol in stocks['Ticker']:\n",
    "    api_url = f'https://sandbox.iexapis.com/stable/stock/{symbol}/quote?token={IEX_CLOUD_API_TOKEN}'\n",
    "    data = requests.get(api_url).json()\n",
    "    final_dataframe = final_dataframe.append(\n",
    "                                        pd.Series([symbol, \n",
    "                                                   data['latestPrice'], \n",
    "                                                   data['marketCap'], \n",
    "                                                   'N/A'], \n",
    "                                                  index = my_columns), \n",
    "                                        ignore_index = True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "final_dataframe"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Using Batch API Calls to Improve Performance\n",
    "\n",
    "Batch API calls are one of the easiest ways to improve the performance of your code.\n",
    "\n",
    "This is because HTTP requests are typically one of the slowest components of a script.\n",
    "\n",
    "Also, API providers will often give you discounted rates for using batch API calls since they are easier for the API provider to respond to.\n",
    "\n",
    "IEX Cloud limits their batch API calls to 100 tickers per request. Still, this reduces the number of API calls we'll make in this section from 500 to 5 - huge improvement! In this section, we'll split our list of stocks into groups of 100 and then make a batch API call for each group."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function sourced from \n",
    "# https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks\n",
    "def chunks(lst, n):\n",
    "    \"\"\"Yield successive n-sized chunks from lst.\"\"\"\n",
    "    for i in range(0, len(lst), n):\n",
    "        yield lst[i:i + n]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "symbol_groups = list(chunks(stocks['Ticker'], 100))\n",
    "symbol_strings = []\n",
    "for i in range(0, len(symbol_groups)):\n",
    "    symbol_strings.append(','.join(symbol_groups[i]))\n",
    "#     print(symbol_strings[i])\n",
    "\n",
    "final_dataframe = pd.DataFrame(columns = my_columns)\n",
    "\n",
    "for symbol_string in symbol_strings:\n",
    "#     print(symbol_strings)\n",
    "    batch_api_call_url = f'https://sandbox.iexapis.com/stable/stock/market/batch/?types=quote&symbols={symbol_string}&token={IEX_CLOUD_API_TOKEN}'\n",
    "    data = requests.get(batch_api_call_url).json()\n",
    "    for symbol in symbol_string.split(','):\n",
    "        final_dataframe = final_dataframe.append(\n",
    "                                        pd.Series([symbol, \n",
    "                                                   data[symbol]['quote']['latestPrice'], \n",
    "                                                   data[symbol]['quote']['marketCap'], \n",
    "                                                   'N/A'], \n",
    "                                                  index = my_columns), \n",
    "                                        ignore_index = True)\n",
    "        \n",
    "    \n",
    "final_dataframe"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Calculating the Number of Shares to Buy\n",
    "\n",
    "As you can see in the DataFrame above, we stil haven't calculated the number of shares of each stock to buy.\n",
    "\n",
    "We'll do that next."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "portfolio_size = input(\"Enter the value of your portfolio:\")\n",
    "\n",
    "try:\n",
    "    val = float(portfolio_size)\n",
    "except ValueError:\n",
    "    print(\"That's not a number! \\n Try again:\")\n",
    "    portfolio_size = input(\"Enter the value of your portfolio:\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "position_size = float(portfolio_size) / len(final_dataframe.index)\n",
    "for i in range(0, len(final_dataframe['Ticker'])-1):\n",
    "    final_dataframe.loc[i, 'Number Of Shares to Buy'] = math.floor(position_size / final_dataframe['Price'][i])\n",
    "final_dataframe"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Formatting Our Excel Output\n",
    "\n",
    "We will be using the XlsxWriter library for Python to create nicely-formatted Excel files.\n",
    "\n",
    "XlsxWriter is an excellent package and offers tons of customization. However, the tradeoff for this is that the library can seem very complicated to new users. Accordingly, this section will be fairly long because I want to do a good job of explaining how XlsxWriter works.\n",
    "\n",
    "### Initializing our XlsxWriter Object"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "writer = pd.ExcelWriter('recommended_trades.xlsx', engine='xlsxwriter')\n",
    "final_dataframe.to_excel(writer, sheet_name='Recommended Trades', index = False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Creating the Formats We'll Need For Our `.xlsx` File\n",
    "\n",
    "Formats include colors, fonts, and also symbols like `%` and `$`. We'll need four main formats for our Excel document:\n",
    "* String format for tickers\n",
    "* \\\\$XX.XX format for stock prices\n",
    "* \\\\$XX,XXX format for market capitalization\n",
    "* Integer format for the number of shares to purchase"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "background_color = '#0a0a23'\n",
    "font_color = '#ffffff'\n",
    "\n",
    "string_format = writer.book.add_format(\n",
    "        {\n",
    "            'font_color': font_color,\n",
    "            'bg_color': background_color,\n",
    "            'border': 1\n",
    "        }\n",
    "    )\n",
    "\n",
    "dollar_format = writer.book.add_format(\n",
    "        {\n",
    "            'num_format':'$0.00',\n",
    "            'font_color': font_color,\n",
    "            'bg_color': background_color,\n",
    "            'border': 1\n",
    "        }\n",
    "    )\n",
    "\n",
    "integer_format = writer.book.add_format(\n",
    "        {\n",
    "            'num_format':'0',\n",
    "            'font_color': font_color,\n",
    "            'bg_color': background_color,\n",
    "            'border': 1\n",
    "        }\n",
    "    )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Applying the Formats to the Columns of Our `.xlsx` File\n",
    "\n",
    "We can use the `set_column` method applied to the `writer.sheets['Recommended Trades']` object to apply formats to specific columns of our spreadsheets.\n",
    "\n",
    "Here's an example:\n",
    "\n",
    "```python\n",
    "writer.sheets['Recommended Trades'].set_column('B:B', #This tells the method to apply the format to column B\n",
    "                     18, #This tells the method to apply a column width of 18 pixels\n",
    "                     string_format #This applies the format 'string_format' to the column\n",
    "                    )\n",
    "```"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# writer.sheets['Recommended Trades'].write('A1', 'Ticker', string_format)\n",
    "# writer.sheets['Recommended Trades'].write('B1', 'Price', string_format)\n",
    "# writer.sheets['Recommended Trades'].write('C1', 'Market Capitalization', string_format)\n",
    "# writer.sheets['Recommended Trades'].write('D1', 'Number Of Shares to Buy', string_format)\n",
    "# writer.sheets['Recommended Trades'].set_column('A:A', 20, string_format)\n",
    "# writer.sheets['Recommended Trades'].set_column('B:B', 20, dollar_format)\n",
    "# writer.sheets['Recommended Trades'].set_column('C:C', 20, dollar_format)\n",
    "# writer.sheets['Recommended Trades'].set_column('D:D', 20, integer_format)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This code works, but it violates the software principle of \"Don't Repeat Yourself\". \n",
    "\n",
    "Let's simplify this by putting it in 2 loops:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "column_formats = { \n",
    "                    'A': ['Ticker', string_format],\n",
    "                    'B': ['Price', dollar_format],\n",
    "                    'C': ['Market Capitalization', dollar_format],\n",
    "                    'D': ['Number of Shares to Buy', integer_format]\n",
    "                    }\n",
    "\n",
    "for column in column_formats.keys():\n",
    "    writer.sheets['Recommended Trades'].set_column(f'{column}:{column}', 20, column_formats[column][1])\n",
    "    writer.sheets['Recommended Trades'].write(f'{column}1', column_formats[column][0], string_format)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Saving Our Excel Output\n",
    "\n",
    "Saving our Excel file is very easy:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "writer.save()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
