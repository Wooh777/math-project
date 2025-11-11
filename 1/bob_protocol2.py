# bob_protocol2.py
import socket, argparse, logging, json, time, random, base64
from math import gcd
from Crypto.Cipher import AES


# ============== JSON TX/RX ==============
def send_json(conn, obj):
    conn.sendall((json.dumps(obj) + "\r\n").encode("utf-8"))  # CRLF


def recv_json_lenient(conn, total_timeout=30.0, max_bytes=1_000_000):
    """개행 유무/조각난 JSON까지 관대하게 복원"""
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

            # 줄 단위
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

            # 개행 없이 중괄호 균형으로 완결 감지
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


# ============== Crypto Utils ==============
def pkcs7_pad(data: bytes, block: int = 16) -> bytes:
    r = len(data) % block
    pad = block - r if r else block
    return data + bytes([pad]) * pad


def pkcs7_unpad(data: bytes, block: int = 16) -> bytes:
    if not data or len(data) % block != 0:
        raise ValueError("invalid padding length")
    pad = data[-1]
    if not (1 <= pad <= block) or data[-pad:] != bytes([pad]) * pad:
        raise ValueError("invalid padding pattern")
    return data[:-pad]


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


def gen_rsa_key_400_500():
    P = primes_in_range()
    p = random.choice(P)
    q = random.choice([x for x in P if x != p])
    n = p * q
    phi = (p - 1) * (q - 1)
    for ec in [3, 5, 17, 257, 65537]:
        if ec < phi and gcd(ec, phi) == 1:
            e = ec
            break
    d = modinv(e, phi)
    return n, e, d


# ============== Handler (Protocol II) ==============
def handle_protocol2(conn, log, server_first=False):
    n, e, d = gen_rsa_key_400_500()
    log(f"[Bob] RSA ready: n={n}, e={e} (d hidden)")

    # (옵션) 서버-먼저 공개키 전송
    if server_first:
        send_json(
            conn, {"opcode": 1, "type": "RSA", "public": e, "parameter": {"n": n}}
        )
        log("[Bob] (server-first) sent RSA public")

    # Alice의 시작 메시지 대기
    start = recv_json_lenient(conn, total_timeout=30.0)
    log(f"[Bob] start from Alice: {start}")
    # RSA 또는 RSAKey 둘 다 트리거로 허용
    if (
        not start
        or start.get("opcode") != 0
        or str(start.get("type", "")).upper() not in ("RSA", "RSAKEY")
    ):
        log("[Bob] unexpected start message, proceeding anyway")

    # Bob → 공개키 전송
    send_json(conn, {"opcode": 1, "type": "RSA", "public": e, "parameter": {"n": n}})
    log("[Bob] sent RSA public (e,n)")

    # Alice → AES 키(바이트별 RSA 암호 리스트) 수신
    msg = recv_json_lenient(conn, total_timeout=30.0)
    if not msg or msg.get("opcode") == 3:
        log(f"[Bob] error or no message: {msg}")
        return
    enc_list = msg.get("encrypted_key")
    if not isinstance(enc_list, list):
        log("[Bob] invalid encrypted_key")
        return

    aes_bytes = bytes([pow(int(c), d, n) for c in enc_list])
    if len(aes_bytes) != 32:
        log(f"[Bob] warning: AES key length {len(aes_bytes)} (expected 32)")
        # 과제는 32바이트를 기대. 안전하게 맞춰줌(잘못된 구현과도 상호작용되도록).
        aes_bytes = (aes_bytes + b"\x00" * 32)[:32]

    # Bob → AES("hello") 전송
    cipher = AES.new(aes_bytes, AES.MODE_ECB)
    ct = cipher.encrypt(pkcs7_pad(b"hello", 16))
    send_json(
        conn,
        {
            "opcode": 2,
            "type": "AES",
            "encryption": base64.b64encode(ct).decode("utf-8"),
        },
    )
    log("[Bob] sent AES ciphertext (hello)")

    # Alice → AES("world") 수신 및 복호화
    msg2 = recv_json_lenient(conn, total_timeout=30.0)
    if not msg2:
        log("[Bob] no AES reply from Alice")
        return
    if msg2.get("opcode") == 3:
        log(f"[Bob] Alice error: {msg2}")
        return
    try:
        ct2 = base64.b64decode(msg2.get("encryption", ""))
        pt2 = pkcs7_unpad(cipher.decrypt(ct2), 16)
        log(f'[Bob] decrypted from Alice: "{pt2.decode("utf-8","ignore")}"')
    except Exception as e:
        log(f"[Bob] AES decode error: {e}")


# ============== Main ==============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("-l", "--log", default="INFO")
    ap.add_argument(
        "--server-first",
        action="store_true",
        help="공개키(e,n)를 Alice 요청 전에 먼저 전송",
    )
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO))

    def log(msg):
        logging.info(msg)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((args.host, args.port))
        srv.listen(1)
        log(f"[Bob] Protocol II server listening on {args.host}:{args.port}")

        conn, addr = srv.accept()
        with conn:
            log(f"[Bob] connection from {addr}")
            handle_protocol2(conn, log, server_first=args.server_first)


if __name__ == "__main__":
    main()
