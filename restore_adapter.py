"""Restore the byte-identical LoRA weights from checked parts (standard library only)."""
from pathlib import Path
import hashlib,json,os

root=Path(__file__).resolve().parent / "released_adapter"
manifest=json.loads((root / "weights_manifest.json").read_text())
dest=root / manifest["filename"]
temp=dest.with_suffix(dest.suffix+".tmp")
hash_all=hashlib.sha256();size=0
try:
    with temp.open("wb") as output:
        for part in manifest["parts"]:
            block=(root / part["path"]).read_bytes()
            if len(block)!=part["bytes"] or hashlib.sha256(block).hexdigest()!=part["sha256"]:
                raise ValueError("Adapter part failed verification: "+part["path"])
            output.write(block);hash_all.update(block);size+=len(block)
    if size!=manifest["bytes"] or hash_all.hexdigest()!=manifest["sha256"]:
        raise ValueError("Complete adapter failed verification")
    os.replace(temp,dest)
except BaseException:
    temp.unlink(missing_ok=True)
    raise
print("Restored and verified:",dest)
