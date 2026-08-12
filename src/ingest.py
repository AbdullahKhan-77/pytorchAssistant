"""
ingest.py - Stage 1: read PyTorch docs from the installed package's docstrings.
Auto-discovers every PUBLIC torch module, extracts each documented callable's
signature + docstring, de-duplicates, and saves to pytorch_docs_raw.json.
Run:  python src/ingest.py
"""

import inspect
import json
import pkgutil
import importlib

import torch


def discover_modules(root_module, root_name):
    """Recursively find every public submodule of a package."""
    discovered = {root_name: root_module}
    if not hasattr(root_module, "__path__"):
        return discovered
    for info in pkgutil.walk_packages(root_module.__path__, prefix=root_name + "."):
        name = info.name
        if any(part.startswith("_") for part in name.split(".")):
            continue
        try:
            mod = importlib.import_module(name)
            discovered[name] = mod
        except Exception:
            continue
    return discovered


def extract_docs(module, module_name):
    """Pull docstring + signature for every public, documented member of a module."""
    docs = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(module, name)
        except Exception:
            continue
        if not callable(obj):
            continue
        doc = inspect.getdoc(obj)
        if not doc or len(doc) < 40:
            continue
        try:
            sig = str(inspect.signature(obj))
        except (ValueError, TypeError):
            sig = ""
        docs.append({
            "id": f"{module_name}.{name}",
            "source": f"{module_name}.{name}",
            "text": f"{module_name}.{name}{sig}\n\n{doc}",
        })
    return docs


def main():
    print("Discovering modules...")
    modules = discover_modules(torch, "torch")
    modules["torch.Tensor"] = torch.Tensor
    print(f"Discovered {len(modules)} modules")

    all_docs = []
    seen_ids = set()
    for mod_name, mod in modules.items():
        for doc in extract_docs(mod, mod_name):
            if doc["id"] not in seen_ids:
                seen_ids.add(doc["id"])
                all_docs.append(doc)

    print(f"Collected {len(all_docs)} unique documented objects")
    if all_docs:
        print("---- sample ----")
        print(all_docs[0]["text"][:400])

    with open("pytorch_docs_raw.json", "w", encoding="utf-8") as f:
        json.dump(all_docs, f, indent=2)
    print(f"\nSaved {len(all_docs)} docs to pytorch_docs_raw.json")


if __name__ == "__main__":
    main()