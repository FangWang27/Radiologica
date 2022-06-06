from getpass import getpass
from click import password_option
import mysql.connector
from mysql.connector import connect, Error
_connection = None

# This models handles the SQL connection with the server 


def get_connection() -> mysql.connector.connection_cext.CMySQLConnection:
    global _connection
    if not _connection or not _connection.is_connected():
        _connection =  mysql.connector.connect(host = 'localhost',
        user = input("Enter user name: "),
        password = getpass('Enter Password: '),
        database = 'Simulates'
    )
    return _connection