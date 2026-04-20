"""
AI 程式生成器 — 本機 Streamlit 工具

流程:輸入需求 → Claude API 生程式碼 → 預覽/編輯 → 一鍵 commit + push 到 GitHub

先備條件 (只做一次):
  1. pip install -r builder_requirements.txt
  2. 設定 Anthropic API Key:  export ANTHROPIC_API_KEY="sk-ant-..."
  3. 首次執行會自動 git clone kuan928/crm 到 ./crm_repo/
     (你的機器需已登入 GitHub,  git config --global user.email / user.name 設好)

使用:
  streamlit run builder.py
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import anthropic
import streamlit as st

REPO_URL = "https://github.com/kuan928/crm.git"
REPO_DIR = Path(__file__).parent / "crm_repo"
MODEL = "claude-opus-4-7"

SYSTEM_PROMPT = """你是資深 Python / Streamlit 工程師,專門生產可直接執行的單檔程式。

你負責的 repo 是訂單/客戶分析儀表板 (kuan928/crm),用 Streamlit 開發。
使用者描述想要的功能,你生成**完整、可立即執行**的 Python 檔案。

規則:
1. 產出單一 .py 檔案,必要時才拆分。除非使用者明確要求多檔。
2. 預設技術棧: streamlit, pandas, numpy, plotly.express, openpyxl。
3. 使用者介面用繁體中文;變數 / 程式邏輯保持英文或中文皆可但要一致。
4. 程式碼必須可以 streamlit run 直接跑,沒有執行時期錯誤。
5. 不要省略 import,不要留 TODO。
6. 需要額外套件時,在回覆末尾用 `## 額外套件` 區塊列出。
7. 回覆格式嚴格如下:

## 檔名
{只寫檔名,例如 customer_rfm.py}

## 說明
{2-4 行簡短說明功能}

## 程式碼
```python
{完整程式碼}
```

## 額外套件
{只列出超出預設技術棧的套件,每行一個。沒有就寫 無}
"""


# ==================== Git helpers ====================

def run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def ensure_repo() -> Path:
    """首次執行 git clone,之後 git pull。"""
    if not REPO_DIR.exists():
        code, out = run(["git", "clone", REPO_URL, str(REPO_DIR)])
        if code != 0:
            raise RuntimeError(f"git clone 失敗: {out}")
    else:
        run(["git", "pull", "--rebase"], cwd=REPO_DIR)
    return REPO_DIR


def git_commit_push(filename: str, commit_msg: str) -> str:
    run(["git", "add", filename], cwd=REPO_DIR)
    code, out = run(["git", "commit", "-m", commit_msg], cwd=REPO_DIR)
    if code != 0 and "nothing to commit" in out:
        return "沒有變更需要提交。"
    code, out = run(["git", "push", "origin", "HEAD"], cwd=REPO_DIR)
    if code != 0:
        return f"❌ push 失敗:\n{out}"
    return f"✅ 已推送到 kuan928/crm\n{out}"


# ==================== Claude API ====================

@st.cache_resource
def client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def generate_code(user_request: str) -> str:
    """呼叫 Claude Opus 4.7,使用 prompt caching 快取 system prompt。"""
    out_box = st.empty()
    collected: list[str] = []

    with client().messages.stream(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_request}],
    ) as stream:
        for text in stream.text_stream:
            collected.append(text)
            out_box.markdown("```\n" + "".join(collected) + "\n```")
        final = stream.get_final_message()

    usage = final.usage
    st.caption(
        f"tokens — input:{usage.input_tokens} "
        f"cache_read:{usage.cache_read_input_tokens} "
        f"cache_write:{usage.cache_creation_input_tokens} "
        f"output:{usage.output_tokens}"
    )
    return "".join(collected)


def parse_response(text: str) -> dict:
    """從模型回覆抽出檔名 / 說明 / 程式碼 / 額外套件。"""
    def section(title: str) -> str:
        m = re.search(rf"##\s*{title}\s*\n(.+?)(?=\n##\s|\Z)", text, re.S)
        return m.group(1).strip() if m else ""

    code_block = re.search(r"```(?:python)?\n(.+?)```", section("程式碼"), re.S)
    return {
        "filename": section("檔名").splitlines()[0].strip() if section("檔名") else "",
        "description": section("說明"),
        "code": code_block.group(1).strip() if code_block else "",
        "packages": section("額外套件"),
    }


# ==================== UI ====================

st.set_page_config(page_title="AI 程式生成器", page_icon="🤖", layout="wide")
st.title("🤖 AI 程式生成器 → kuan928/crm")
st.caption(f"輸入功能需求 → Claude {MODEL} 生成 → 一鍵推送到 GitHub")

# 檢查 API Key
if not os.environ.get("ANTHROPIC_API_KEY"):
    st.error("請先設定環境變數 ANTHROPIC_API_KEY,然後重新啟動此程式。")
    st.code('export ANTHROPIC_API_KEY="sk-ant-..."', language="bash")
    st.stop()

# 確認 repo 存在
with st.spinner("同步 GitHub repo..."):
    try:
        ensure_repo()
    except Exception as e:
        st.error(f"repo 初始化失敗:{e}")
        st.stop()
st.success(f"repo 已就緒:{REPO_DIR}")

# --- 輸入 ---
st.subheader("① 描述你要的功能")
user_req = st.text_area(
    "越具體越好,可以包含:輸入資料格式、欄位、想看的圖表、分析邏輯…",
    height=140,
    placeholder="例:客戶 RFM 分析儀表板。上傳訂單 CSV,自動算每位客戶的 Recency / Frequency / Monetary,分 5 級,顯示客群熱圖與前 20 名高價值客戶。",
)

if st.button("🚀 生成程式", type="primary", disabled=not user_req.strip()):
    with st.spinner("Claude 思考中..."):
        raw = generate_code(user_req)
    st.session_state["last_raw"] = raw
    st.session_state["parsed"] = parse_response(raw)

# --- 預覽 / 編輯 ---
parsed = st.session_state.get("parsed")
if parsed:
    st.subheader("② 預覽 / 編輯")
    if not parsed["code"]:
        st.warning("沒解析到程式碼區塊。下方是原始輸出,請手動處理。")
        st.code(st.session_state.get("last_raw", ""))
        st.stop()

    c1, c2 = st.columns([1, 2])
    with c1:
        filename = st.text_input("檔名", value=parsed["filename"] or "new_feature.py")
        st.markdown("**說明**")
        st.info(parsed["description"] or "(無)")
        if parsed["packages"] and parsed["packages"].strip() != "無":
            st.markdown("**需加的額外套件**")
            st.code(parsed["packages"])
    with c2:
        code = st.text_area("程式碼", value=parsed["code"], height=500)

    st.subheader("③ 提交到 GitHub")
    commit_msg = st.text_input("commit 訊息", value=f"新增 {filename}")
    if st.button("📤 寫入檔案並 push"):
        target = REPO_DIR / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code, encoding="utf-8")
        st.toast(f"已寫入 {target}", icon="💾")
        with st.spinner("git commit + push..."):
            msg = git_commit_push(filename, commit_msg)
        if msg.startswith("✅"):
            st.success(msg)
            st.balloons()
            st.markdown(f"→ <https://github.com/kuan928/crm/blob/main/{filename}>")
        else:
            st.error(msg)
