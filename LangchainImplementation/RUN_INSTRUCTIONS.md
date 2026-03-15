# 🚀 Run Commands (Sequential Order)

You must use **TWO separate terminals** because the server needs to stay running while you run the tests.

---

## 🟢 Terminal 1: Application Server
*This terminal will run the server. Keep it open!*

**Step 1. Navigate to the folder**
```powershell
cd c:\Users\Dhruv\OneDrive\Desktop\MPR1\Multi-Agent-System-for-Adaptive-Production-Scheduling\LangchainImplementation
```

**Step 2. Activate Virtual Environment**
```powershell
& .\.venv\Scripts\Activate.ps1
```

**Step 3. Start the Server**
```powershell
python -m a2a.server
```

---

## 🔵 Terminal 2: Tests & Client
*Open a NEW PowerShell window for this.*

**Step 1. Navigate to the folder**
```powershell
cd c:\Users\Dhruv\OneDrive\Desktop\MPR1\Multi-Agent-System-for-Adaptive-Production-Scheduling\LangchainImplementation
```

**Step 2. Activate Virtual Environment**
```powershell
& .\.venv\Scripts\Activate.ps1
```

**Step 3. Run the Client Verification** (Fast check)
```powershell
python -m a2a.example
```

**Step 4. Run the Full System Simulation** (Deep check)
```powershell
python test_full_system.py
```

---

## Micro LM (Optional, offline)

Run a tiny (>=1000 parameter) local language model instead of GPT-2 for `MachineAgent.llm`:

```powershell
$env:USE_MICRO_LM="1"
python test_full_system.py
```
