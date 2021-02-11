import sqlite3

conn = sqlite3.connect('database.db')

c = conn.cursor()

# c.execute('''CREATE TABLE saved_logs (ID INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT, y_value TEXT)''')

for row in c.execute("SELECT * from saved_logs;"):
    print(row)
