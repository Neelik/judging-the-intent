import time

from peewee import (
    CharField,
    ForeignKeyField,
    IntegerField,
    Model,
    TextField,
    TimestampField,
)

from judging_the_intent.db import DATABASE


class BaseModel(Model):
    class Meta:
        database = DATABASE


class Query(BaseModel):
    q_id = CharField(primary_key=True)
    dataset_name = CharField()
    text = TextField()


class Intent(BaseModel):
    i_id = CharField()
    query = ForeignKeyField(Query, backref="intents")
    text = TextField()


class Document(BaseModel):
    d_id = CharField(primary_key=True)
    text = TextField()


class Triple(BaseModel):
    query = ForeignKeyField(Query, backref="triples")
    intent = ForeignKeyField(Intent, backref="triples", null=True)
    document = ForeignKeyField(Document, backref="triples")


class Config(BaseModel):
    model_name = CharField()
    version = CharField()

    class Meta:
        indexes = ((("model_name", "version"), True),)


class Annotation(BaseModel):
    triple = ForeignKeyField(Triple, backref="annotations")
    config = ForeignKeyField(Config, backref="annotations")
    result = IntegerField(null=True)
    error = TextField(null=True)
    timestamp = TimestampField(default=time.time)

    class Meta:
        indexes = ((("triple", "config"), True),)
