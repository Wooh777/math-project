#!/usr/bin/env python3
import sys, json, math, base64, glob, os
from Crypto.Cipher import AES


# ---------- commons ----------
def read_json_lines(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # 라인에 잡다한 로그가 섞여 있으면 무시
                continue


def pkcs7_unpad(data: bytes, block=16):
    if not data or len(data) % block != 0:
        raise ValueError("invalid padding length")
    pad = data[-1]
    if not (1 <= pad <= block):
        raise ValueError("invalid padding value")
    if data[-pad:] != bytes([pad]) * pad:
        raise ValueError("invalid padding pattern")
    return data[:-pad]


def egcd(a, b):
    if b == 0:
        return (a, 1, 0)
    g, x, y = egcd(b, a % b)
    return (g, y, x - (a // b) * y)


def modinv(a, m):
    g, x, _ = egcd(a, m)
    if g != 1:
        raise ValueError("no inverse")
    return x % m


def trial_factor(n):
    if n % 2 == 0:
        return 2, n // 2
    f = 3
    lim = int(math.isqrt(n)) + 1
    while f <= lim:
        if n % f == 0:
            return f, n // f
        f += 2
    return None, None


# ---------- Protocol II ----------
def decrypt_protocol2(objs):
    e = n = None
    enc_list = None
    aes_msgs = []
    for o in objs:
        t = str(o.get("type", "")).upper()
        if o.get("opcode") == 1 and t == "RSA":
            e = int(o["public"])
            n = int(o["parameter"]["n"])
        elif o.get("opcode") == 2 and t == "RSA" and "encrypted_key" in o:
            enc_list = [int(x) for x in o["encrypted_key"]]
        elif o.get("opcode") == 2 and t == "AES" and "encryption" in o:
            aes_msgs.append(o["encryption"])
    if e is None or n is None or enc_list is None:
        return None  # not protocol II

    p, q = trial_factor(n)
    if not p or not q:
        raise RuntimeError(f"[P2] factoring failed for n={n}")
    phi = (p - 1) * (q - 1)
    d = modinv(e, phi)
    key_bytes = bytes([pow(c, d, n) for c in enc_list])
    if len(key_bytes) < 32:
        key_bytes = (key_bytes + b"\x00" * 32)[:32]
    else:
        key_bytes = key_bytes[:32]

    out = {
        "protocol": "II",
        "rsa": {"n": n, "e": e, "p": p, "q": q, "d": d},
        "aes_key_hex": key_bytes.hex(),
        "plaintexts": [],
    }
    if aes_msgs:
        cipher = AES.new(key_bytes, AES.MODE_ECB)
        for b64 in aes_msgs:
            ct = base64.b64decode(b64)
            pt = pkcs7_unpad(cipher.decrypt(ct))
            try:
                out["plaintexts"].append(pt.decode("utf-8"))
            except:
                out["plaintexts"].append(pt.hex())
    return out


# ---------- Protocol III ----------
def decrypt_protocol3(objs):
    # 수집
    p = g = None
    publics = []
    aes_msgs = []
    for o in objs:
        t = str(o.get("type", "")).upper()
        if t == "DH":
            if "parameter" in o:
                prm = o["parameter"]
                if "p" in prm:
                    p = int(prm["p"])
                if "g" in prm:
                    g = int(prm["g"])
            if "public" in o:
                publics.append(int(o["public"]))
        if (
            o.get("opcode") == 2
            and str(o.get("type", "")).upper() == "AES"
            and "encryption" in o
        ):
            aes_msgs.append(o["encryption"])
    if not p or not g or len(publics) < 2:
        return None  # not protocol III

    A, B = publics[0], publics[1]

    # 이산로그 브루트포스 (p<=500 조건)
    def solve_secret(pub):
        for x in range(2, p - 1):
            if pow(g, x, p) == pub:
                return x
        return None

    a = solve_secret(A)
    if a is None:
        # swap 시도
        a = solve_secret(B)
        A, B = B, A
        if a is None:
            raise RuntimeError("[P3] discrete log failed")

    s = pow(B, a, p)  # 공유비밀
    s_bytes = s.to_bytes(2, byteorder="big")
    aes_key = (s_bytes * (32 // len(s_bytes) + 1))[:32]

    out = {
        "protocol": "III",
        "dh": {"p": p, "g": g, "A": A, "B": B, "a": a, "shared": s},
        "aes_key_hex": aes_key.hex(),
        "plaintexts": [],
    }
    if aes_msgs:
        cipher = AES.new(aes_key, AES.MODE_ECB)
        for b64 in aes_msgs:
            ct = base64.b64decode(b64)
            pt = pkcs7_unpad(cipher.decrypt(ct))
            try:
                out["plaintexts"].append(pt.decode("utf-8"))
            except:
                out["plaintexts"].append(pt.hex())
    return out


# ---------- batch ----------
def process_file(path):
    objs = list(read_json_lines(path))
    res = decrypt_protocol2(objs)
    if res is None:
        res = decrypt_protocol3(objs)
    if res is None:
        return {"file": path, "error": "Unknown or insufficient data"}
    res["file"] = path
    return res


def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_decrypt_logs.py <files or globs>")
        print("  e.g., python batch_decrypt_logs.py adv_protocol_*.log")
        sys.exit(1)

    files = []
    for arg in sys.argv[1:]:
        matched = glob.glob(arg)
        if matched:
            files.extend(matched)
        else:
            files.append(arg)

    for f in sorted(files):
        try:
            r = process_file(f)
            if "error" in r:
                print(f"\n=== {f} ===\nERROR: {r['error']}")
                continue
            print(f"\n=== {r['file']} (Protocol {r['protocol']}) ===")
            if r["protocol"] == "II":
                R = r["rsa"]
                print(f"n={R['n']}, e={R['e']}, p={R['p']}, q={R['q']}, d={R['d']}")
            else:
                D = r["dh"]
                print(
                    f"p={D['p']}, g={D['g']}, A={D['A']}, B={D['B']}, a={D['a']}, shared={D['shared']}"
                )
            print("AES key (hex):", r["aes_key_hex"])
            if r["plaintexts"]:
                print("Plaintexts:")
                for i, pt in enumerate(r["plaintexts"], 1):
                    print(f"  [{i}] {pt}")
            else:
                print("No AES ciphertexts found.")
        except Exception as e:
            print(f"\n=== {f} ===\nERROR: {e}")


if __name__ == "__main__":
    main()
