#!/usr/bin/env python3
import sqlite3

with sqlite3.connect('podcasts.sqlite') as conn:
    cursor = conn.cursor()
    for table_name in ['show', 'episode', 'line']:
        print(table_name)
        for doof in cursor.execute("select * from " + table_name):
            print(doof)
        input("Press enter to continue")
        print("------------------")


