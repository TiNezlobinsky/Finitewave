import os
import ast
from pathlib import Path
import pandas as pd
from collections import defaultdict


def find_classes_in_file(file_path):
    """Extract class names from a given Python file."""
    with open(file_path, 'r') as file:
        node = ast.parse(file.read())
        return [n.name for n in node.body if isinstance(n, ast.ClassDef)]


def generate_api_pages(base_path):
    base_path = Path(base_path).parent
    api_dir = base_path.joinpath('docs/api')
    api_dir.mkdir(parents=True, exist_ok=True)

    modules = []
    submodules = []
    class_names = []
    md_files = []

    for root, dirs, files in os.walk(base_path.joinpath('finitewave')):

        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                path_to_class = Path(root, file)
                try:
                    path_to_class.relative_to(base_path).with_suffix('')
                except ValueError:
                    print(f"Skipping {path_to_class}")
                    continue

                for class_name in find_classes_in_file(path_to_class):

                    path_to_class = path_to_class.relative_to(base_path).with_suffix('')
                    class_module = f"{path_to_class.as_posix().replace('/', '.')}.{class_name}"

                    api_module_path = Path(*path_to_class.parts[:3])
                    path_to_md = api_dir.joinpath(api_module_path, path_to_class.stem).with_suffix('.md')
                    path_to_md.parent.mkdir(parents=True, exist_ok=True)

                    with path_to_md.open('w') as f:
                        f.write(f"# {class_name}\n")
                        f.write(f"::: {class_module}\n")

                    class_names.append(class_name)
                    md_files.append(path_to_md.relative_to(api_dir).as_posix())
                    modules.append(path_to_class.parts[1])
                    submodules.append(path_to_class.parts[2])

    df = pd.DataFrame({'module': modules, 'submodule': submodules,
                       'class': class_names, 'md_file': md_files})

    visited_modules = []
    api_section = "# API Reference\n\n"
    for (module, submodule), group in df.groupby(['module', 'submodule']):

        if module not in visited_modules:
            visited_modules.append(module)
            api_section += f"## {module}\n\n"

        if module != 'tools':
            api_section += f"### {submodule}\n\n"

        group = group.sort_values(by='class')
        for i, row in group.iterrows():
            api_section += f"- [{row['class']}](./api/{row['md_file']})\n"
        api_section += "\n"

    return api_section


    # # Generate the API section content with nested structure
    # api_section = "# API Reference\n\n"
    # # for folder, classes in sorted(structure.items()):            
    #     api_section += f"## {folder}\n\n"
    #     for class_name, md_file in classes:
    #         api_section += f"- [{class_name}](./api/{md_file})\n"
    #     api_section += "\n"

    # return api_section


# Generate the API section
path = Path(__file__).parents[1].joinpath('finitewave')
api_md_content = generate_api_pages(path)

# Write to docs/api.md
with open('docs/api.md', 'w') as f:
    f.write(api_md_content)
