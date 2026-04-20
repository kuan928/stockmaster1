# AI 程式生成器 (builder.py)

一個跑在你**自己電腦**的 Streamlit 工具:輸入功能描述 → Claude 產生程式碼 → 一鍵 push 到 `kuan928/crm`。

---

## 一、首次安裝 (只做一次)

### 1. 下載檔案到你的電腦
把 `builder.py`、`builder_requirements.txt` 這兩個檔案存到一個資料夾,例如 `~/crm-builder/`。

### 2. 安裝套件
```bash
cd ~/crm-builder
pip install -r builder_requirements.txt
```

### 3. 設定 Anthropic API Key
到 <https://console.anthropic.com> 申請 API Key,然後:

**macOS / Linux**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```
(想要每次自動載入,可以寫進 `~/.zshrc` 或 `~/.bashrc`)

**Windows (PowerShell)**
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

### 4. 確認 git 可以 push 到 kuan928/crm
```bash
git config --global user.name "你的名字"
git config --global user.email "你的 email"
```
第一次 push 時 git 會要你登入 GitHub (瀏覽器授權或 PAT)。

---

## 二、每次使用

```bash
cd ~/crm-builder
streamlit run builder.py
```

瀏覽器會自動開啟 `http://localhost:8501`,接著:

1. **描述功能** — 例如:「客戶 RFM 分析儀表板。上傳訂單 CSV,自動算 Recency / Frequency / Monetary 分數,顯示熱圖。」
2. **按「生成程式」** — Claude 會邊想邊寫,大約 10-30 秒。
3. **預覽 / 微調程式碼** — 不滿意可以在文字框直接改。
4. **按「寫入檔案並 push」** — 自動 commit 並 push 到 `kuan928/crm`。
5. 畫面會給你 GitHub 連結,點開就看到新檔案。

---

## 三、部署到永久網址

push 成功後,到 <https://share.streamlit.io>:
- Repository: `kuan928/crm`
- Branch: `main`
- Main file path: `你剛生成的檔名.py`
- Deploy → 得到 `https://xxx.streamlit.app`

之後每次 push 新檔,Streamlit Cloud 會自動重新部署。

---

## 常見問題

**Q: push 失敗說 permission denied?**
A: 你的 git 沒登入 GitHub。跑一次 `git pull` 在 `crm_repo/` 裡面觸發登入,或設定 PAT:
```
git remote set-url origin https://<user>:<token>@github.com/kuan928/crm.git
```

**Q: 生出來的程式不能跑?**
A: 把錯誤訊息貼回需求欄位,按一次「生成程式」,Claude 會修正。

**Q: 要換成比較便宜的模型?**
A: 編輯 `builder.py` 裡的 `MODEL = "claude-opus-4-7"`,改成 `claude-sonnet-4-6` 或 `claude-haiku-4-5`。
