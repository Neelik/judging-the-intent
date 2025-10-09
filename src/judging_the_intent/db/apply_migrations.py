from pathlib import Path
import importlib.util

def apply_migrations():
    migration_directory = Path(__file__).parent.joinpath("db", "migrations")
    migration_list = Path(migration_directory).glob('**/*.py')
    for migration_path in migration_list:
        # because path is object not string
        path_in_str = str(migration_path)
        spec = importlib.util.spec_from_file_location(migration_path.name.split(".")[0], path_in_str)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.jti_migrate()

if __name__ == "__main__":
    apply_migrations()