import argparse
import zipfile
from pathlib import Path


def add_path(zf: zipfile.ZipFile, path: Path, arc_prefix: str) -> None:
    if path.is_file():
        zf.write(path, f"{arc_prefix}/{path.name}")
        return
    for p in sorted(path.rglob("*")):
        if p.is_file():
            zf.write(p, f"{arc_prefix}/{p.relative_to(path)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-id", required=True, help="学号")
    parser.add_argument("--name", required=True, help="姓名")
    parser.add_argument("--outdir", default=".", help="输出目录")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    zip_name = f"{args.student_id}{args.name}作业1.zip"
    zip_path = outdir / zip_name

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        add_path(zf, root / "src", "src")
        add_path(zf, root / "outputs", "outputs")
        add_path(zf, root / "report.md", ".")
        add_path(zf, root / "README.md", ".")
        add_path(zf, root / "requirements.txt", ".")

    print(f"Created: {zip_path}")


if __name__ == "__main__":
    main()

