from playhouse.migrate import *
from judging_the_intent.db import DATABASE

def jti_migrate():
    migrator = PostgresqlMigrator(database=DATABASE)
    explanation_field = TextField(null=True)

    try:
        with DATABASE.atomic():
            migrate(migrator.add_column('annotation', 'explanation', explanation_field))
    except (IntegrityError, ProgrammingError):
        pass

if __name__ == "__main__":
    jti_migrate()
