from playhouse.migrate import *
from judging_the_intent.db import DATABASE
import logging
LOGGER = logging.getLogger(__file__)

def jti_migrate():
    migrator = PostgresqlMigrator(database=DATABASE)
    fine_tuned = BooleanField(default=True)
    intent_aware = BooleanField(default=False)

    try:
        with DATABASE.atomic():
            migrate(
                # Drop the original index
                migrator.drop_index('config', 'config_model_name_version'),

                # Add the fields
                migrator.add_column('config', 'fine_tuned', fine_tuned),
                migrator.add_column('config', 'with_intent', intent_aware),

                # Add unique index using new fields
                migrator.add_index('config', ('model_name', 'version', 'fine_tuned', 'with_intent'), True),

            )
    except (IntegrityError, ProgrammingError):
        LOGGER.info("Migration failed...")
        pass

if __name__ == "__main__":
    jti_migrate()
