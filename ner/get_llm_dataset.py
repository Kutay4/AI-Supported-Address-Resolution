import asyncio, aiohttp, aiofiles, anyio, json, logging
import pandas as pd
from tenacity import retry, wait_random_exponential, stop_after_attempt
from sklearn.model_selection import train_test_split

DEEP_INFRA_TOKEN = "Enter Deep infra token here"

# ────────── LOG SETTINGS ──────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("addr-extractor")

#────────── CONSTANTS ──────────────
SYSTEM_TEXT = """Extract pk(post code), diger(any identifier), mahalle(neighbourhood) name, sokak(street) name,
semt(district),subregion (ilçe) name, region (il) name as json.
Leave null if info missing. Don't fix anything. Leave it as it is.
Adresses will be in Turkish and only 1 address will be sent to you always.
Do not mix any irrelevant field in mahalle,sokak,semt,il,ilce. Always write irrelevant fields to diger.
Requested json format: {\"mahalle\":\"neighbourhood name\",\"sokak\":\"street name/cadde name\",\"semt\":\"district name\",\"il\":\"region name\",\"ilce\":\"subregion name\",
\"diger\":\"apt names,apt no,floor no,nearby places etc\",\"pk\":\"post code\"}"""
MAX_CONCURRENCY = 200
OUTFILE = "llmdatasetnew.jsonl"

#Dataset Configuration comes here. Just Be sure that you give a series that consists of addresses.
#Stratify shuffle to get addresses from all labels.
DATA = pd.read_csv("dataset/train.csv")
x_train,x_test,y_train,y_test = train_test_split(DATA["address"],DATA["label"],train_size=200_000, shuffle=True, stratify=DATA["label"],random_state=42)
TOTAL = len(x_train)

limiter = anyio.CapacityLimiter(MAX_CONCURRENCY)
addr_q: asyncio.Queue[str | None] = asyncio.Queue()
write_q: asyncio.Queue[dict | None] = asyncio.Queue()

# ────────── LLM CALL ───────────
@retry(wait=wait_random_exponential(1, 20), stop=stop_after_attempt(6))
async def call_llm(session: aiohttp.ClientSession, msg: str) -> str:
    headers = {
        "content-type": "application/json",
        "Authorization": f"Bearer {DEEP_INFRA_TOKEN}",
    }
    payload = {
       # "model":"openai/gpt-oss-20b",
        "model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "messages": [
            {"role": "system", "content": SYSTEM_TEXT},
            {"role": "user", "content": msg},
        ],
    }
    async with limiter:
        async with session.post(
            "https://api.deepinfra.com/v1/openai/chat/completions",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status != 200:
                raise aiohttp.ClientResponseError(
                    resp.request_info, (), status=resp.status
                )
            data = await resp.json()
            return data["choices"][0]["message"]["content"]

# ────────── WORKER ────────────────
async def worker(session: aiohttp.ClientSession):
    while True:
        addr = await addr_q.get()
        if addr is None:
            addr_q.task_done()
            break
        try:
            answer = await call_llm(session, addr)
            await write_q.put({addr: answer})
        except Exception as e:
            log.warning("Address could not be processed: %s | Error: %r", addr, e)
        finally:
            addr_q.task_done()

# ────────── WRITER + PROGRESS ─────
async def writer():
    processed = 0
    async with aiofiles.open(OUTFILE, "a", encoding="utf-8") as f:
        while True:
            item = await write_q.get()
            if item is None:
                write_q.task_done()
                break
            await f.write(json.dumps(item, ensure_ascii=False) + "\n")
            await f.flush()
            processed += 1
            if processed % 100 == 0 or processed == TOTAL:
                log.info("Saved: %d / %d", processed, TOTAL)
            write_q.task_done()

# ────────── MAIN FLOW ──────────────
async def main():
    for adr in x_train:
        addr_q.put_nowait(adr)
    for _ in range(MAX_CONCURRENCY):
        addr_q.put_nowait(None)

    log.info("Total addresses: %d | Concurrency: %d", TOTAL, MAX_CONCURRENCY)

    async with aiohttp.ClientSession() as session:
        async with asyncio.TaskGroup() as tg:
            for _ in range(MAX_CONCURRENCY):
                tg.create_task(worker(session))

            tg.create_task(writer())

    log.info("🚀 All tasks completed, exiting.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("⏹️  Stopped by user (Ctrl-C)")
