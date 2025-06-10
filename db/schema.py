from peewee import CharField, ForeignKeyField, Model

from db import DATABASE


class BaseModel(Model):
    class Meta:
        database = DATABASE


class Query(BaseModel):
    q_id = CharField(primary_key=True)
    dataset_name = CharField()
    text = CharField()


class Intent(BaseModel):
    i_id = CharField()
    query = ForeignKeyField(Query, backref="intents")
    text = CharField()


class Document(BaseModel):
    d_id = CharField(primary_key=True)
    text = CharField()


class Triple(BaseModel):
    query = ForeignKeyField(Query, backref="triples")
    intent = ForeignKeyField(Intent, backref="triples", null=True)
    document = ForeignKeyField(Document, backref="triples")


class LLM(BaseModel):
    name = CharField()


class Annotation(BaseModel):
    triple = ForeignKeyField(Triple, backref="annotations")
    llm = ForeignKeyField(LLM, backref="annotations")
