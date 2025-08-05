import sqlite3

def get_connection():
    return sqlite3.connect("chatbot.db")

def create_table():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT
        )
    """)
    conn.commit()
    conn.close()

def add_log(question, answer):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO logs (question, answer) VALUES (?, ?)", (question, answer))
    conn.commit()
    conn.close()

def get_logs():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM logs")
    rows = cursor.fetchall()
    conn.close()
    return rows

if __name__ == "__main__":
    create_table()
