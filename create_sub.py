import re
from pathlib import Path

# 重複防止のためのセット
included_headers = set()


def expand_file(filepath: Path) -> str:
    content_lines = []
    filepath = filepath.resolve()
    if filepath in included_headers:
        return f"// Skipped: {filepath.name} already included\n"

    included_headers.add(filepath)

    with filepath.open() as f:
        for line in f:
            include_match = re.match(r'#include\s+"(.+)"', line)
            if include_match:
                included_path = (filepath.parent / include_match.group(1)).resolve()
                # content_lines.append(f"// Begin include: {included_path.name}\n")
                content_lines.append(expand_file(included_path))
                # content_lines.append(f"// End include: {included_path.name}\n")
            elif line.strip() == "#pragma once":
                continue  # 無視
            else:
                content_lines.append(line)
    return "".join(content_lines)


def inline_main_cpp(main_cpp_path: str, output_path: str):
    main_path = Path(main_cpp_path)
    out_lines = []
    with main_path.open() as f:
        for line in f:
            include_match = re.match(r'#include\s+"(.+)"', line)
            if include_match:
                included_path = (main_path.parent / include_match.group(1)).resolve()
                # out_lines.append(f"// Begin include: {included_path.name}\n")
                out_lines.append(expand_file(included_path))
                # out_lines.append(f"// End include: {included_path.name}\n")
            else:
                out_lines.append(line)

    with open(output_path, "w") as out_file:
        out_file.writelines(out_lines)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Inline C++ main file with included headers.")
    parser.add_argument("--i", type=str, help="Path to the main C++ file.", default="main.cpp")
    parser.add_argument("--o", type=str, help="Path to the output file.", default="sub.cpp")
    args = parser.parse_args()
    inline_main_cpp(args.i, args.o)
