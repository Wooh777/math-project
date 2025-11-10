import socket, argparse, logging, json, random, base64
from Crypto.Cipher import AES


# ====== PKCS#7 pad / unpad ======
def pkcs7_pad(data: bytes, block_size: int = 16) -> bytes:
    rem = len(data) % block_size
    pad_len = block_size - rem if rem != 0 else block_size
    return data + bytes([pad_len]) * pad_len


def pkcs7_unpad(data: bytes, block_size: int = 16) -> bytes:
    if not data or len(data) % block_size != 0:
        raise ValueError("invalid padding length")
    pad_len = data[-1]
    if not (1 <= pad_len <= block_size):
        raise ValueError("invalid padding value")
    if data[-pad_len:] != bytes([pad_len]) * pad_len:
        raise ValueError("invalid padding pattern")
    return data[:-pad_len]


# ====== number utils ======
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


def prime_factors(n: int):
    fac = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            fac.add(d)
            n //= d
        d += 1
    if n > 1:
        fac.add(n)
    return fac


def is_generator(g: int, p: int) -> bool:
    if not is_prime(p):
        return False
    order = p - 1
    for q in prime_factors(order):
        if pow(g, order // q, p) == 1:
            return False
    return True


# ====== net utils ======
def recv_line(sock, timeout=10.0):
    sock.settimeout(timeout)
    buf = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            return buf.decode("utf-8") if buf else ""
        buf += chunk
        if b"\n" in buf:
            line, _ = buf.split(b"\n", 1)
            return line.decode("utf-8")


def send_json(sock, obj):
    sock.sendall((json.dumps(obj) + "\n").encode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--addr", required=True)
    ap.add_argument("-p", "--port", required=True, type=int)
    ap.add_argument("-l", "--log", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO))

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.settimeout(10.0)
        s.connect((args.addr, args.port))
        logging.info(f"[Alice] connected to {args.addr}:{args.port}")

        # 1) DH 시작 요청
        send_json(s, {"opcode": 0, "type": "DH"})

        # 2) Bob의 DH 파라미터 수신 (p,g,B)
        line = recv_line(s)
        if not line:
            raise RuntimeError("no DH params from Bob")
        resp1 = json.loads(line)
        if resp1.get("opcode") == 3:
            raise RuntimeError(f"Bob error: {resp1.get('error')}")
        if not (resp1.get("opcode") == 1 and resp1.get("type") == "DH"):
            raise RuntimeError("invalid DH params format")

        p = int(resp1["parameter"]["p"])
        g = int(resp1["parameter"]["g"])
        B = int(resp1["public"])
        logging.info(f"[Alice] DH params <- p={p}, g={g}, B={B}")

        # 2-1) 검증: p 소수, g 생성자
        if not is_prime(p):
            send_json(s, {"opcode": 3, "error": "incorrect prime number"})
            raise RuntimeError("p is not prime")
        if not is_generator(g, p):
            send_json(s, {"opcode": 3, "error": "incorrect generator"})
            raise RuntimeError("g is not a generator")

        # 3) Alice 비밀키 a, 공개키 A 생성 후 전송
        a = random.randint(2, p - 2)
        A = pow(g, a, p)
        send_json(s, {"opcode": 1, "type": "DH", "public": A})

        # 4) 공유 비밀키 s = B^a mod p → AES 키 파생
        s_val = pow(B, a, p)
        s_bytes = s_val.to_bytes(2, byteorder="big")
        aes_key = (s_bytes * (32 // len(s_bytes)))[:32]
        logging.info("[Alice] shared secret derived (32 bytes)")

        # 5) Bob이 보낸 "hello" 수신 & 복호
        line = recv_line(s)
        if not line:
            raise RuntimeError("no AES message from Bob")
        resp2 = json.loads(line)
        if resp2.get("opcode") == 3:
            raise RuntimeError(f"Bob error: {resp2.get('error')}")
        cipher = AES.new(aes_key, AES.MODE_ECB)
        ct = base64.b64decode(resp2["encryption"])
        pt = pkcs7_unpad(cipher.decrypt(ct), 16)
        logging.info(f'[Alice] Decrypted from Bob: "{pt.decode()}"')

        # 6) "world" 암호화해서 Bob에게 전송
        ct2 = cipher.encrypt(pkcs7_pad(b"world", 16))
        b64 = base64.b64encode(ct2).decode("utf-8")
        send_json(s, {"opcode": 2, "type": "AES", "encryption": b64})
        logging.info("[Alice] sent AES ciphertext (world)")

    except Exception as e:
        logging.exception(f"[Alice] error: {e}")
    finally:
        try:
            s.close()
        except:
            pass


if __name__ == "__main__":
    main()
