# tools.py
import sqlite3
import json

def query_database(query):
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    return json.dumps(rows)

'''
def query_data():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM processed_data')
    rows = cursor.fetchall()
    conn.close()
    return rows

def format_for_agent(data):
    formatted_data = []
    for row in data:
        formatted_data.append({
            "filename": row[1],
            "url": row[2],
            "chat_history": json.loads(row[3])
        })
    return formatted_data
'''