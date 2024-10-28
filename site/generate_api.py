import os
import ast
from pathlib import Path


def find_classes_in_file(file_path):
    """Extract class names from a given Python file."""
    with open(file_path, 'r') as file:
        node = ast.parse(file.read())
        return [n.name for n in node.body if isinstance(n, ast.ClassDef)]


def generate_api_pages(base_path):
    base_path = Path(base_path).parent
    api_section = "# API Reference\n\n"
    api_dir = base_path.joinpath('docs/api')
    api_dir.mkdir(parents=True, exist_ok=True)

    for root, dirs, files in os.walk(base_path.joinpath('finitewave')):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                module_path = Path(root) / file
                class_names = find_classes_in_file(module_path)

                for class_name in class_names:

                    class_module = module_path.relative_to(base_path)
                    class_module = class_module.with_suffix('')
                    class_module = class_module.as_posix().replace('/', '.')
                    class_module = f"{class_module}.{class_name}"
  
                    md_file_name = api_dir.joinpath(f"{class_name}.md")

                    # Create a separate Markdown file for each class
                    with md_file_name.open('w') as f:
                        f.write(f"# {class_name}\n")
                        f.write(f"::: {class_module}\n")

                    api_section += f"- [{class_name}](./api/{md_file_name.name})\n"

    return api_section


# Generate the API section
path = Path(__file__).parents[1].joinpath('finitewave')
api_md_content = generate_api_pages(path)

# Write to docs/api.md
with open('docs/api.md', 'w') as f:
    f.write(api_md_content)
