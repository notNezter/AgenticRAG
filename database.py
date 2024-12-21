# database.py
import sqlite3
import json
from pydantic import BaseModel

class ProcessedData(BaseModel):
    filename: str
    url: str
    chat_history: list

def init_db():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_data (
            id INTEGER PRIMARY KEY,
            filename TEXT,
            url TEXT,
            chat_history TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_data(filename: str, url: str, chat_history: list):
    """
    Inserts a basic record of a file with chat history into the database.

    Args:
        filename (str): The name of the uploaded file.
        url (str): The file's URL or storage path.
        chat_history (list): The conversation history.
    """
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    try:
        chat_history_str = json.dumps(chat_history) if isinstance(chat_history, list) else chat_history
        cursor.execute('''
            INSERT INTO processed_data (filename, url, chat_history)
            VALUES (?, ?, ?)
        ''', (filename, url, chat_history_str))
        conn.commit()
    finally:
        conn.close()

def insert_processed_file(filename: str, file_path: str, chat_history: list):
    """
    Inserts a detailed record of a processed file, including chat history.

    Args:
        filename (str): The name of the file.
        file_path (str): The file's storage path.
        chat_history (list): Detailed conversation history (e.g., speaker, message, response).
    """
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    try:
        chat_history_json = json.dumps(chat_history)
        cursor.execute('''
            INSERT INTO processed_data (filename, url, chat_history)
            VALUES (?, ?, ?)
        ''', (filename, file_path, chat_history_json))
        conn.commit()
    finally:
        conn.close()

def insert_chat_history(conversation_id, speaker, message):
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO chat_history (conversation_id, speaker, message)
            VALUES (?, ?, ?)
        ''', (conversation_id, speaker, message))
        conn.commit()
    finally:
        conn.close()
        
def query_data():
    """
    Fetches all basic records from the database.

    Returns:
        list[dict]: A list of records with filename, URL, and chat history.
    """
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT filename, url, chat_history FROM processed_data')
        rows = cursor.fetchall()
        return [{"filename": row[0], "url": row[1], "chat_history": json.loads(row[2])} for row in rows]
    finally:
        conn.close()

def query_processed_files() -> list[ProcessedData]:
    """
    Fetches detailed records and maps them to `ProcessedData`.

    Returns:
        list[ProcessedData]: A list of detailed processed files.
    """
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT filename, url, chat_history FROM processed_data')
        rows = cursor.fetchall()
        return [ProcessedData(
            filename=row[0],
            url=row[1],
            chat_history=json.loads(row[2])
        ) for row in rows]
    finally:
        conn.close()

def format_for_agent(data):
    formatted_data = []
    for row in data:
        formatted_data.append({
            "filename": row[1],
            "url": row[2],
            "chat_history": json.loads(row[3])
        })
    return formatted_data