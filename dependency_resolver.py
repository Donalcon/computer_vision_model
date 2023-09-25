import subprocess


def get_child_packages(target, pip_tree_lines):
    child_packages = []
    is_target_section = False
    for line in pip_tree_lines:
        if target in line:
            is_target_section = True
        elif is_target_section and not line.startswith(" "):
            break
        elif is_target_section and line.startswith("│"):
            continue
        elif is_target_section and line.startswith("├──") or line.startswith("└──"):
            child_package = line.split(" ")[1].split("[")[0]
            child_packages.append(child_package)
    return child_packages


def get_all_related_packages(target, pip_tree_lines, visited=None):
    if visited is None:
        visited = set()

    if target in visited:
        return []

    visited.add(target)
    related_packages = []

    child_packages = get_child_packages(target, pip_tree_lines)
    related_packages.extend(child_packages)
    for child_package in child_packages:
        subchild_packages = get_all_related_packages(child_package, pip_tree_lines, visited)
        related_packages.extend(subchild_packages)

    return related_packages


if __name__ == "__main__":

    target_packages = []

    target_packages2 = ["sphinxcontrib-applehelp", "sphinxcontrib-devhelp", "sphinxcontrib-htmlhelp","sphinxcontrib-jsmath",
                        "sphinxcontrib-qthelp", "sphinxcontrib-serializinghtml", "setuptools-scm", "py-cpuinfo", "pyasn1",
                        "pyDeprecate", "pyreadline3", "pyrsistent", "oauthlib", "mpmath", "MarkupSafe", "markdown-it-py"
                       "google-auth-oauthlib", "future", "flatbuffers", "colorama", "commonmark", "docutils", "humanfriendly",
                       "boto3", "Babel", "attrs", "alabaster"]


    target_packages1 = ["super-gradients", "boto-3", "botocore", "jmespath", "python-dateutil", "six", "urllib3",
                       "s3transfer", "coverage", "Deprecated", "wrapt", "einops", "hydra-core", "antlr4-python3-runtime",
                       "omegaconf", "packaging", "json-tricks", "jsonschema", "contourpy", "fonttools", "kiwisolver",
                       "protobuf", "onnx", "onnx-simplifier", "rich", "onnxruntime", "coloredlogs", "sympy", "pip-tools",
                       "build", "psutil", "pycocotools", "Pygments", "pyparsing", "rapidfuzz", "setuptools", "Sphinx",
                       "Jinja2", "requests", "snowballstemmer", "sphinx-rtd-theme ", "sphinxcontrib-jquery", "imagesize",
                       "stringcase", "tensorboard", "absl-py", "google-auth", "cachetools", "pyasn1-modules", "rsa",
                       "six", "requests-oauthlib", "certifi", "charset-normalizer", "idna", "grpcio", "Markdown", "zipp",
                       "tensorboard-data-server", "Werkzeug", "wheel", "termcolor", "torchmetrics", "torch", "treelib"]

    # Obtain the pipdeptree output
    result = subprocess.run(['pipdeptree'], stdout=subprocess.PIPE)
    pip_tree_output = result.stdout.decode('utf-8')
    pip_tree_lines = pip_tree_output.strip().split('\n')

    for target_package in target_packages:
        # Identify all related packages including child, sub-child, etc.
        related_packages = get_all_related_packages(target_package, pip_tree_lines)

        # Uninstall target and all related packages
        packages_to_uninstall = [target_package] + related_packages
        if packages_to_uninstall:
            print(f"Uninstalling {target_package} and its related packages...")
            uninstall_command = ["pip", "uninstall", "-y"] + packages_to_uninstall
            subprocess.run(uninstall_command)
        else:
            print(f"{target_package} is not installed or has no related packages.")
