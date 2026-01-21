from pathlib import Path

def clean_labels(dataset_path):
    labels_dir = Path(dataset_path) / "labels"

    for txt in labels_dir.rglob("*.txt"):
        lines = txt.read_text().strip().splitlines()
        mouse_only = [
            ln for ln in lines
            if ln.strip() and ln.split()[0] == "0"
        ]
        txt.write_text("\n".join(mouse_only) + ("\n" if mouse_only else ""))

# Run for both datasets
clean_labels("dataset_original")
clean_labels("dataset_enhanced")

print("Done: cat labels removed, mouse only kept.")
