#!/usr/bin/env python3
import curses
import sys
import time
import locale

from curses.textpad import Textbox, rectangle

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB

from joblib import load


# Sets locale to user default
locale.setlocale(locale.LC_ALL, '')

# Get used preferred encoding
code = locale.getpreferredencoding()

#stdscr = curses.initscr()

def test(stdscr):
    # Initialize screen

    # Setting up for functionality
    # No echoing of keystrokes on screen
    curses.noecho()

    # React to keys instantly
    curses.cbreak()

    # Handle special keystrokes
    stdscr.keypad(True)

    time.sleep(5)
    curses.flash()

    # Terminating application
    curses.nocbreak()
    stdscr.keypad(False)
    curses.echo()
    curses.endwin()

def main(stdscr):
    stdscr.clear()
    headerwin = curses.newwin(2, curses.COLS, 0, 0)
    headerwin.addstr("Welcome to SENTIMENT_ANALYZER_PROGRAM!\n")
    headerwin.addstr("Please input a string of text for analysis:\n")

    #inp = stdscr.getstr()
    #stdscr.addstr(inp)

    textbox = curses.textpad.Textbox(stdscr)

    stdscr.refresh()

    while True:
        textbox.edit()


if __name__ == '__main__':
    #curses.wrapper(main)
    #test(stdscr)
    print(curses.KEY_BACKSPACE)
    print(ord('^?'))
