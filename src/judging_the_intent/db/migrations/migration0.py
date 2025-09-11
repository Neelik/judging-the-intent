from playhouse.migrate import *
from judging_the_intent.db import DATABASE

if __name__ == "__main__":
    migrator = PostgresqlMigrator(database=DATABASE)
    truncated_field = BooleanField(default=False)

    with DATABASE.atomic():
        migrate(migrator.add_column('annotation', 'truncated', truncated_field))
