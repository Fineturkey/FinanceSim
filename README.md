# FinanceSim
Imported data from the python library yfinance (Yahoo Finance) and used three indicators (RSI, SMA, and EMA) to back track the returns over time

You can run with just python, there are libarires that you need to install (pandas, yfinance, numpy, matplotlib, time, os), otherwise run normally
python finalProject.py

Uses parallel processing (threading and processors) and does a time comparaison with sequential. The point of this code isn't really to back track a trading algorithm but rather benchmarking to see the time taken between sequential and parallel. 

In this case it's done in python (more overhead in general) and each task isn't really worth doing it in parallel... also not using a well known parallel technique (point jumping, euler tour, etc etc) so it makes sense why it takes so long.
