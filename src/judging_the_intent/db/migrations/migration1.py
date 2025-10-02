from playhouse.migrate import *
from judging_the_intent.db import DATABASE

if __name__ == "__main__":
    migrator = PostgresqlMigrator(database=DATABASE)
    explanation_field = TextField(null=True)

    with DATABASE.atomic():
        migrate(migrator.add_column('annotation', 'explanation', explanation_field))
