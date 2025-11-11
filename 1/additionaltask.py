#!/usr/bin/env python3
import sys, json, math, base64
from Crypto.Cipher import AES


# ----------------- 유틸 -----------------
def is_probable_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def trial_factor(n):
    """작은 n(예: 과제 범위) 에 대한 단순 브루트포스 소인수분해"""
    if n % 2 == 0:
        return 2, n // 2
    f = 3
    limit = int(math.isqrt(n)) + 1
    while f <= limit:
        if n % f == 0:
            return f, n // f
        f += 2
    return None, None


def egcd(a, b):
    if b == 0:
        return (a, 1, 0)
    g, x, y = egcd(b, a % b)
    return (g, y, x - (a // b) * y)


def modinv(a, m):
    g, x, _ = egcd(a, m)
    if g != 1:
        raise ValueError("modular inverse does not exist")
    return x % m


def pkcs7_unpad(data: bytes, block=16):
    if not data or len(data) % block != 0:
        raise ValueError("invalid padding length")
    pad = data[-1]
    if pad < 1 or pad > block:
        raise ValueError("invalid padding value")
    if data[-pad:] != bytes([pad]) * pad:
        raise ValueError("invalid padding pattern")
    return data[:-pad]


# ----------------- 로직 -----------------
def decrypt_from_logs(lines):
    # parse newline-separated JSON snippets (lenient)
    objs = []
    for L in lines:
        L = L.strip()
        if not L:
            continue
        try:
            objs.append(json.loads(L))
        except Exception:
            # ignore non-json lines
            continue

    # extract RSA public (e,n), encrypted_key list, and AES ciphertexts
    e = None
    n = None
    enc_list = None
    aes_ciphertexts = []  # base64 strings
    for obj in objs:
        if obj.get("opcode") == 1 and str(obj.get("type", "")).upper() == "RSA":
            e = int(obj.get("public"))
            param = obj.get("parameter", {})
            if "n" in param:
                n = int(param["n"])
        if obj.get("opcode") == 2 and str(obj.get("type", "")).upper() == "RSA":
            # some variants may use opcode 2 for encrypted key (defensive)
            if "encrypted_key" in obj:
                enc_list = obj["encrypted_key"]
        if obj.get("opcode") == 2 and str(obj.get("type", "")).upper() == "AES":
            if "encryption" in obj:
                aes_ciphertexts.append(obj["encryption"])
        # also consider field name differences
        if "encrypted_key" in obj and enc_list is None:
            enc_list = obj["encrypted_key"]

    if e is None or n is None:
        raise RuntimeError("Could not find RSA public (e,n) in logs.")
    if enc_list is None:
        raise RuntimeError("Could not find encrypted_key list in logs.")

    # Factor n (assumes small semiprime as in assignment)
    p, q = trial_factor(n)
    if not p or not q:
        raise RuntimeError(f"Failed to factor n={n} with simple trial division.")
    if not is_probable_prime(p) or not is_probable_prime(q):
        raise RuntimeError(f"Found factors but they are not prime: p={p}, q={q}")

    phi = (p - 1) * (q - 1)
    # careful modular inverse
    d = modinv(e, phi)

    # decrypt each element in encrypted_key: expecting per-byte RSA encryption pow(byte,e,n)
    plaintext_bytes = bytearray()
    for c in enc_list:
        # ensure integer
        ci = int(c)
        m = pow(ci, d, n)  # should produce original byte (0-255)
        if m < 0 or m > 255:
            # If assignment encoded larger chunks, handle accordingly (but unlikely here)
            raise RuntimeError(f"Decrypted integer {m} out of byte range")
        plaintext_bytes.append(m)

    # normalize AES key length to 32 (if shorter, warn; if longer, truncate)
    if len(plaintext_bytes) < 32:
        print(
            "[WARN] Recovered AES key length < 32 bytes; padding with zeros",
            file=sys.stderr,
        )
        aes_key = (bytes(plaintext_bytes) + b"\x00" * 32)[:32]
    else:
        aes_key = bytes(plaintext_bytes[:32])

    results = {
        "rsa": {"n": n, "e": e, "p": p, "q": q, "d": d},
        "aes_key_hex": aes_key.hex(),
        "decrypted_aes_plaintexts": [],
    }

    # decrypt any AES ciphertexts found
    if aes_ciphertexts:
        cipher = AES.new(aes_key, AES.MODE_ECB)
        for b64 in aes_ciphertexts:
            try:
                ct = base64.b64decode(b64)
                pt = pkcs7_unpad(cipher.decrypt(ct))
                try:
                    pt_str = pt.decode("utf-8")
                except:
                    pt_str = repr(pt)
                results["decrypted_aes_plaintexts"].append(pt_str)
            except Exception as e:
                results["decrypted_aes_plaintexts"].append(f"[DECRYPTION FAILED: {e}]")
    return results


# ----------------- CLI -----------------
def main():
    if len(sys.argv) >= 2 and sys.argv[1] != "-":
        fname = sys.argv[1]
        with open(fname, "r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    try:
        res = decrypt_from_logs(lines)
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(1)

    # 출력
    print("=== RSA params (recovered) ===")
    for k, v in res["rsa"].items():
        print(f"{k}: {v}")
    print()
    print("AES key (hex):", res["aes_key_hex"])
    print()
    if res["decrypted_aes_plaintexts"]:
        print("=== Decrypted AES plaintexts ===")
        for i, pt in enumerate(res["decrypted_aes_plaintexts"], 1):
            print(f"[{i}] {pt}")
    else:
        print("No AES ciphertexts found in logs.")


if __name__ == "__main__":
    main()
