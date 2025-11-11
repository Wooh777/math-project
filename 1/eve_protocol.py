# eve_protocol2.py
import socket, argparse, logging, json, time, base64
from Crypto.Cipher import AES


# ============== JSON TX/RX (관대한 파서) ==============
def send_json(conn, obj):
    conn.sendall((json.dumps(obj) + "\r\n").encode("utf-8"))


def recv_json_lenient(conn, total_timeout=30.0, max_bytes=1_000_000):
    deadline = time.time() + total_timeout
    buf = b""
    braces = 0
    started = False
    conn.settimeout(1.0)
    while time.time() < deadline:
        try:
            chunk = conn.recv(4096)
            if not chunk:
                if not buf.strip():
                    return None
                t = buf.decode("utf-8", "ignore").strip()
                i, j = t.find("{"), t.rfind("}")
                if i != -1 and j != -1 and j > i:
                    return json.loads(t[i : j + 1])
                return None

            buf += chunk
            if len(buf) > max_bytes:
                raise RuntimeError("response too large")

            # line-first
            while b"\n" in buf or b"\r\n" in buf:
                line, sep, rest = buf.partition(b"\n")
                if sep == b"":
                    line, sep, rest = buf.partition(b"\r\n")
                buf = rest
                t = line.decode("utf-8", "ignore").strip()
                if not t:
                    continue
                i, j = t.find("{"), t.rfind("}")
                if i != -1 and j != -1 and j > i:
                    return json.loads(t[i : j + 1])

            # no newlines: brace-balance
            for bch in chunk:
                if bch == ord("{"):
                    braces += 1
                    started = True
                elif bch == ord("}"):
                    braces -= 1
                    if started and braces == 0:
                        t = buf.decode("utf-8", "ignore")
                        i, j = t.find("{"), t.rfind("}")
                        if i != -1 and j != -1 and j > i:
                            return json.loads(t[i : j + 1])
        except socket.timeout:
            continue
    return None


# ============== 수론/암호 유틸 ==============
def is_prime(n: int) -> bool:
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


def primes_in_range(lo=400, hi=500):
    return [n for n in range(lo, hi + 1) if is_prime(n)]


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


def pkcs7_unpad(data: bytes, block: int = 16) -> bytes:
    if not data or len(data) % block != 0:
        raise ValueError("invalid padding length")
    pad = data[-1]
    if not (1 <= pad <= block) or data[-pad:] != bytes([pad]) * pad:
        raise ValueError("invalid padding")
    return data[:-pad]


# ============== Eve (MITM proxy for Protocol II) ==============
def run_proxy(listen_host, listen_port, bob_host, bob_port):
    logging.info(
        f"[Eve] listening on {listen_host}:{listen_port}, forwarding to {bob_host}:{bob_port}"
    )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ls:
        ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ls.bind((listen_host, listen_port))
        ls.listen(1)
        a_conn, a_addr = ls.accept()

    with a_conn:
        logging.info(f"[Eve] Alice connected from {a_addr}")

        # Eve connects to Bob
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as b_conn:
            b_conn.connect((bob_host, bob_port))
            logging.info(f"[Eve] connected to Bob at {bob_host}:{bob_port}")

            # 1) Alice -> start (opcode 0, type RSA)
            a_start = recv_json_lenient(a_conn, total_timeout=30.0)
            logging.info(f"[Eve] A→ start: {a_start}")
            if a_start:
                send_json(b_conn, a_start)

            # 2) Bob -> public (opcode 1, type RSA, public e, parameter n)
            b_pub = recv_json_lenient(b_conn, total_timeout=30.0)
            logging.info(f"[Eve] B→ pub: {b_pub}")
            # 중계
            if b_pub:
                send_json(a_conn, b_pub)

            # Eve: e, n 확보 → p,q 소인수분해 → d 계산
            e = int(b_pub.get("public"))
            n = int(b_pub.get("parameter", {}).get("n"))
            # n은 400~500 사이 소수의 곱 → 브루트포스 분해 가능
            P = primes_in_range()
            p = q = None
            for x in P:
                if n % x == 0 and (n // x) in P:
                    p, q = x, n // x
                    break
            if not p or not q:
                raise RuntimeError("[Eve] factoring failed")
            phi = (p - 1) * (q - 1)
            d = modinv(e, phi)
            logging.info(f"[Eve] factored n: p={p}, q={q}, φ={phi}, d={d}")

            # 3) Alice -> RSA encrypted_key(list) : 중계 + 복호로 AES 키 획득
            a_key = recv_json_lenient(a_conn, total_timeout=30.0)
            logging.info(f"[Eve] A→ enc_key: {str(a_key)[:200]}")
            if a_key:
                send_json(b_conn, a_key)
            enc_list = a_key.get("encrypted_key", [])
            aes_bytes = bytes([pow(int(c), d, n) for c in enc_list])
            if len(aes_bytes) != 32:
                logging.warning(
                    f"[Eve] AES key length {len(aes_bytes)} (expected 32) — padding to 32"
                )
                aes_bytes = (aes_bytes + b"\x00" * 32)[:32]
            logging.info(f"[Eve] recovered AES-256 key: {aes_bytes.hex()}")

            cipher = AES.new(aes_bytes, AES.MODE_ECB)

            # 4) Bob -> AES("hello"): 중계 + 복호 출력
            b_msg = recv_json_lenient(b_conn, total_timeout=30.0)
            logging.info(f"[Eve] B→ AES: {b_msg}")
            if b_msg:
                send_json(a_conn, b_msg)
            try:
                ct1 = base64.b64decode(b_msg.get("encryption", ""))
                pt1 = pkcs7_unpad(cipher.decrypt(ct1), 16).decode("utf-8", "ignore")
                logging.info(f'[Eve] decrypted B→A plaintext: "{pt1}"')
            except Exception as e2:
                logging.warning(f"[Eve] decrypt B→A failed: {e2}")

            # 5) Alice -> AES("world"): 중계 + 복호 출력
            a_msg = recv_json_lenient(a_conn, total_timeout=30.0)
            logging.info(f"[Eve] A→ AES: {a_msg}")
            if a_msg:
                send_json(b_conn, a_msg)
            try:
                ct2 = base64.b64decode(a_msg.get("encryption", ""))
                pt2 = pkcs7_unpad(cipher.decrypt(ct2), 16).decode("utf-8", "ignore")
                logging.info(f'[Eve] decrypted A→B plaintext: "{pt2}"')
            except Exception as e3:
                logging.warning(f"[Eve] decrypt A→B failed: {e3}")

            logging.info("[Eve] passive interception complete.")


if __name__ == "__main__":
    import argparse, logging

    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="127.0.0.1")
    ap.add_argument("--lport", type=int, required=True)
    ap.add_argument("--to", dest="to_host", default="127.0.0.1")
    ap.add_argument("--tport", type=int, required=True)
    ap.add_argument("-l", "--log", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO))
    run_proxy(args.listen, args.lport, args.to_host, args.tport)
