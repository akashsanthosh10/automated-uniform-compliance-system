""""import sqlite3

# Connect to the database (or create it if it doesn't exist)
conn = sqlite3.connect('recognition_data.db')

# Create a cursor object to execute SQL commands
cursor = conn.cursor()

with conn:
    conn.execute('''
                CREATE TABLE IF NOT EXISTS person (
                    name TEXT PRIMARY KEY,
                    id TEXT NOT NULL,
                    email TEXT,
                    gender TEXT,
                    dob TEXT
                )
            ''')

# Commit the changes and close the connection
conn.commit()
conn.close()"""
import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('recognition_data.db')

# Retrieve contents of the person table
with conn:
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM fine_details')
    rows = cursor.fetchall()

# Print the contents of the person table
for row in rows:
    print(row)

def delete():
    conn = sqlite3.connect('recognition_data.db')

    # Create a cursor object to execute SQL queries
    cursor = conn.cursor()

    # Execute the DELETE statement to remove all rows from the fine_details table
    cursor.execute('DELETE FROM fine_details')

    # Commit the transaction to save the changes
    conn.commit()

    # Close the cursor and the connection
    cursor.close()
    conn.close()
delete()
"""import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('recognition_data.db')

# Create a cursor object to execute SQL queries
cursor = conn.cursor()

# Execute the DELETE statement to remove all rows from the fine_details table
cursor.execute('DELETE FROM fine_details')

# Commit the transaction to save the changes
conn.commit()

# Close the cursor and the connection
cursor.close()
conn.close()"""
