import time
from typing import Type
from peewee import (
    BooleanField,
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


class Dataset(BaseModel):
    name = CharField(primary_key=True)


class Query(BaseModel):
    q_id = CharField()
    dataset_name = ForeignKeyField(Dataset, backref="queries")
    text = TextField()

    class Meta:
        indexes = ((("q_id", "dataset_name"), True),)


class Intent(BaseModel):
    i_id = CharField()
    query = ForeignKeyField(Query, backref="intents")
    text = TextField()
    source = CharField(default="human")


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
    fine_tuned = BooleanField(default=True)
    with_intent = BooleanField(default=False)
    prompt_style = CharField(choices=["human", "human-intent", "binary", "binary-intent", "dna", "dna-intent"], default="binary")

    class Meta:
        indexes = ((("model_name", "version", "fine_tuned", "with_intent", "prompt_style"), True),)


class Annotation(BaseModel):
    triple = ForeignKeyField(Triple, backref="annotations")
    config = ForeignKeyField(Config, backref="annotations")
    result = IntegerField(null=True)
    error = TextField(null=True)
    timestamp = TimestampField(default=time.time)
    truncated = BooleanField(default=False)
    explanation = TextField(null=True)

    class Meta:
        indexes = ((("triple", "config"), True),)
