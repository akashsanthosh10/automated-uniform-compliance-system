"""import sqlite3

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
    
persons = [
    ('agney', 'vml01', 'agney@gmail.com', 'Male', '1990-01-01'),
    ('akash', 'vml02', 'akash@gmail.com', 'Male', '1995-05-15'),
    ('alen', 'vml03', 'alen@gmail.com', 'Male', '1992-05-25'),
    ('shizin', 'vml04', 'shizin@gmail.com', 'Male', '1995-12-15')
    # Add more rows as needed
]

# Insert multiple rows into the person table
with conn:
    conn.executemany('''
        INSERT INTO person (name, id, email, gender, dob)
        VALUES (?, ?, ?, ?, ?)
    ''', persons)

# Commit the changes and close the connection
conn.commit()
conn.close()"""
import smtplib, ssl
from email.message import EmailMessage
#from app2 import password
port = 465  # For SSL
smtp_server = "smtp.gmail.com"
sender_email = "pythonprojectvjec@gmail.com"  # Enter your address
receiver_email = "akashlab246@gmail.com"  # Enter receiver address
password = "xiwqcmqrylfilrfe"
Subject="Hello"
body = """\
Subject: Hi there

This message is sent from Python."""
em=EmailMessage()
em['From']=sender_email
em['To']=receiver_email
em['Subject']=Subject
em.set_content(body)
context = ssl.create_default_context()
with smtplib.SMTP_SSL(smtp_server, port,context=context) as smtp:
    smtp.login(sender_email, password)
    smtp.sendmail(sender_email, receiver_email, em.as_string())
    print("Success")

"""server=smtplib.SMTP("smtp.gmail.com",587)
server.starttls()
server.login(sender_email,"xiwqcmqrylfilrfe")
server.sendmail(sender_email,receiver_email,body)"""