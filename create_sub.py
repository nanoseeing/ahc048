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


# 例: main.cpp を処理して main_flattened.cpp を生成
inline_main_cpp("main.cpp", "sub.cpp")
