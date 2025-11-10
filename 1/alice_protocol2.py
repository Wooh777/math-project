import socket, argparse, logging, json, os, base64
from Crypto.Cipher import AES


# ===== PKCS#7 pad/unpad (no Crypto.Util.Padding) =====
def pkcs7_pad(data: bytes, block_size: int = 16) -> bytes:
    rem = len(data) % block_size
    pad_len = block_size - rem if rem != 0 else block_size
    return data + bytes([pad_len]) * pad_len


def pkcs7_unpad(data: bytes, block_size: int = 16) -> bytes:
    if not data or len(data) % block_size != 0:
        raise ValueError("invalid padding: length")
    pad_len = data[-1]
    if not (1 <= pad_len <= block_size):
        raise ValueError("invalid padding: value")
    if data[-pad_len:] != bytes([pad_len]) * pad_len:
        raise ValueError("invalid padding: pattern")
    return data[:-pad_len]


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

        # 1) RSA 공개키 요청
        req0 = {"opcode": 0, "type": "RSA"}
        send_json(s, req0)

        line = recv_line(s)
        if not line:
            raise RuntimeError("no response for RSA pubkey")
        resp1 = json.loads(line)
        if resp1.get("opcode") == 3:
            raise RuntimeError(f"Bob error: {resp1.get('error')}")
        e = int(resp1["public"])
        n = int(resp1["parameter"]["n"])
        logging.info(f"[Alice] received RSA pubkey e={e}, n={n}")

        # 2) AES 키 생성 후, RSA(e,n)로 바이트 단위 암호화
        aes_key = os.urandom(32)  # 256-bit
        enc_list = [pow(b, e, n) for b in aes_key]
        send_json(s, {"opcode": 2, "type": "RSA", "encrypted_key": enc_list})
        logging.info("[Alice] sent RSA-encrypted AES key")

        # 3) Bob → AES("hello") 수신 & 복호
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

        # 4) "world" 암호화해서 전송
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
