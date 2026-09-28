# -*- encoding: utf-8 -*-
"""清理 fork 中继承自上游的分支，只保留指定分支（默认 main）。

复用 push_via_api.py 的 GH 客户端（gh token + 沙箱代理，只走 api.github.com）。

用法：
    .venv/Scripts/python.exe cleanup_branches.py --dry-run      # 只列出将删除哪些
    .venv/Scripts/python.exe cleanup_branches.py               # 执行删除
"""
import argparse
import sys
import urllib.parse

from push_via_api import DEFAULT_REPO, GH, gh_token


def list_branches(gh: GH) -> list[str]:
    names: list[str] = []
    page = 1
    while True:
        chunk = gh.api(f"/branches?per_page=100&page={page}")
        if not chunk:
            break
        names += [b["name"] for b in chunk]
        if len(chunk) < 100:
            break
        page += 1
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="删除 fork 中除保留分支外的所有分支")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--keep", default="main", help="保留的分支，默认 main")
    parser.add_argument("--dry-run", action="store_true", help="只列出，不删除")
    args = parser.parse_args()

    keep = {b.strip() for b in args.keep.split(",") if b.strip()}
    gh = GH(args.repo, gh_token())

    branches = list_branches(gh)
    targets = [b for b in branches if b not in keep]
    print(f"仓库 {args.repo}：共 {len(branches)} 个分支，保留 {sorted(keep)}，待删除 {len(targets)} 个")

    if args.dry_run:
        for name in targets[:20]:
            print(f"  - {name}")
        if len(targets) > 20:
            print(f"  ... 其余 {len(targets) - 20} 个")
        return

    ok = fail = 0
    for i, name in enumerate(targets, 1):
        ref = urllib.parse.quote(name, safe="")
        try:
            gh.api(f"/git/refs/heads/{ref}", "DELETE")
            ok += 1
        except RuntimeError as exc:
            fail += 1
            print(f"  失败 {name}: {str(exc).splitlines()[0]}")
        if i % 50 == 0 or i == len(targets):
            print(f"  进度 {i}/{len(targets)}（成功 {ok}，失败 {fail}）")

    left = list_branches(gh)
    print(f"\n完成：删除 {ok} 个，失败 {fail} 个；仓库现有分支 {left}")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
