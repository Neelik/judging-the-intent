from playhouse.migrate import *
from judging_the_intent.db import DATABASE
import logging
LOGGER = logging.getLogger(__file__)

def jti_migrate():
    migrator = PostgresqlMigrator(database=DATABASE)
    source = CharField(default="human")

    try:
        with DATABASE.atomic():
            migrate(
                # Add the fields
                migrator.add_column('intent', 'source', source),
            )
    except (IntegrityError, ProgrammingError) as e:
            pass

if __name__ == "__main__":
    jti_migrate()
