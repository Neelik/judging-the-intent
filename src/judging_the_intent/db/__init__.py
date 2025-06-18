import os

from peewee import PostgresqlDatabase

DATABASE = PostgresqlDatabase(
    os.environ.get("POSTGRES_DB"),
    user=os.environ.get("POSTGRES_USER"),
    password=os.environ.get("POSTGRES_PASSWORD"),
    host=os.environ.get("POSTGRES_HOST"),
    port=os.environ.get("POSTGRES_PORT"),
)
