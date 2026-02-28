from pathlib import Path
import importlib.util
import logging
LOGGER = logging.getLogger(__file__)

def apply_migrations():
    migration_directory = Path(__file__).parent.joinpath("migrations")
    migration_list = Path(migration_directory).glob('**/*.py')
    for migration_path in migration_list:
        # because path is object not string
        LOGGER.info(f"\tMigration path: {migration_path}")
        path_in_str = str(migration_path)
        spec = importlib.util.spec_from_file_location(migration_path.name.split(".")[0], path_in_str)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.jti_migrate()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("\tApplying migrations...")
    apply_migrations()